import re
import pdfplumber
from pathlib import Path
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from collections import defaultdict
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pathlib import Path
from langchain.chains.summarize import load_summarize_chain

# Load environment variables
load_dotenv()

# initialize Parser
json_parser = JsonOutputParser()


PDF_PATH = Path("saksoft_q4.pdf")
persist_path = "faiss_index"


def extract_lines(pdf_path: Path) -> List[str]:
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1.5, y_tolerance=3)
            if text:
                # split on hard-returns, strip extra white-space
                lines += [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines


def clean_name(name: str) -> str:
    name = name.upper()
    name = re.sub(r"\b(MR|MS|MRS|DR)\.?\s+", "", name)
    name = re.sub(r"\s{2,}", " ", name)
    return name.strip()


def build_chunks(all_lines: List[str], roles: Dict[str, List[str]]):
    intro = []
    qna_chunks = []

    # Normalize all names in roles
    management = set(clean_name(name) for name in roles.get("management", []))
    management.add("management")
    moderator = set(clean_name(name) for name in roles.get("moderator", []))

    moderator.add("moderator")

    current_q = None
    current_a = []
    collecting_intro = True
    expecting_answer = False

    last_speaker = None
    last_role = None

    is_converstatoin_start = False

    speaker_pattern = re.compile(r"^([A-Z][A-Za-z .&']+):\s*(.*)")

    for line in all_lines:
        line = line.strip()
        if not line:
            continue

        match = speaker_pattern.match(line)

        if match:
            speaker, text = match.groups()
            speaker = speaker.strip()
            text = text.strip()

            norm_spk = clean_name(speaker)

            if not is_converstatoin_start:
                if norm_spk in moderator or norm_spk in management:
                    is_converstatoin_start = True

            if not is_converstatoin_start:
                continue
            # -- Q&A section --
            if norm_spk not in management:
                # New participant question
                if current_q:
                    qna_chunks.append(
                        {
                            **current_q,
                            "answer_speaker": ", ".join({spk for spk, _ in current_a}),
                            "answer_text": " ".join(txt for _, txt in current_a),
                        }
                    )
                current_q = {"question_speaker": speaker, "question_text": text}
                current_a = []
                expecting_answer = True
            else:
                # Management answer
                if expecting_answer:
                    current_a.append((speaker, text))
                    expecting_answer = False  # first answer line received
                else:
                    # May be follow-up answer
                    current_a.append((speaker, text))

            last_speaker = speaker
            if norm_spk in management:
                last_role = "management"
            elif norm_spk in moderator:
                last_role = "moderator"
            else:
                last_role = "participant"
        else:
            if not is_converstatoin_start:
                continue
            if expecting_answer and current_q:
                # Continuation of question
                current_q["question_text"] += " " + line
            elif current_a:
                # Continuation of answer
                last_spk, prev = current_a[-1]
                current_a[-1] = (last_spk, prev + " " + line)

    # Final flush
    if current_q:
        qna_chunks.append(
            {
                **current_q,
                "answer_speaker": ", ".join({spk for spk, _ in current_a}),
                "answer_text": " ".join(txt for _, txt in current_a),
            }
        )

    return {"qna": qna_chunks, "roles": roles}


def prepare_documents(qna_chunks):
    documents = []
    for item in qna_chunks:
        metadata = {
            "question_speaker": item["question_speaker"],
            "answer_speaker": item["answer_speaker"],
        }
        content = f"Q: {item['question_text']}\nA: {item['answer_text']}"
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def remove_think_blocks(text: str) -> str:
    # This pattern removes everything between <think> and </think>, including the tags
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},  # or "cuda"
    encode_kwargs={"normalize_embeddings": True},
)


def create_and_store_embeddings(documents, persist_path="faiss_index"):

    if Path(persist_path).exists():
        print(f"Loading existing FAISS index from {persist_path}")
        # Load existing FAISS index
        vector_db = FAISS.load_local(
            persist_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
        return vector_db

    # Create FAISS index
    vector_db = FAISS.from_documents(documents, embedding_model)

    # Save FAISS index to disk
    vector_db.save_local(persist_path)
    print(f"Vector DB saved to {persist_path}")
    return vector_db


if __name__ == "__main__":
    # Extract lines from PDF
    lines = extract_lines(PDF_PATH)

    # Join first 200 lines for processing
    joined_lines = "\n".join(lines[:200])

    # Initialize LLM chain
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
    prompt = ChatPromptTemplate(
        [
            """
                Here is the first page of an earnings call transcript:

                {text}

                From this, extract the list of **management members** and the **moderator**. 
                For each person, provide:

                - Full name            
                - Role (management/moderator)

                Respond in JSON format as a list of objects like:
                [
                {{
                    "name": "John Doe",                    
                    "role": "management"
                }}
                ]
                """
        ]
    )

    chain = prompt | llm | json_parser

    # Invoke the chain with the joined lines
    json_str = chain.invoke({"text": joined_lines})

    # Parse the JSON output
    result = defaultdict(list)
    for person in json_str:
        result[person["role"]].append(person["name"])

    # Convert defaultdict to normal dict
    result = dict(result)

    # Build chunks from the extracted lines and roles
    chunks = build_chunks(lines, result)

    # Prepare documents from the Q&A chunks
    documents = prepare_documents(chunks["qna"])

    # Create and store embeddings in FAISS
    vector_db = create_and_store_embeddings(documents)

    retriever = vector_db.as_retriever()

    prompt_template = PromptTemplate.from_template(
        """
        You are an assistant that answers questions based on the transcript of an earnings call.

        Use ONLY the context below to answer the question.

        Context:
        {context}

        Question:
        {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt_template)

    qa_chain = create_retrieval_chain(retriever, document_chain)

    query = "any quereis on revenue target for next year?"
    response = qa_chain.invoke({"input": query})
    cleaned_response = remove_think_blocks(response["answer"])

    chain = load_summarize_chain(llm, chain_type="map_reduce")

    summary = chain.invoke(retriever.get_relevant_documents(query))

    print(f"Cleaned Response: {cleaned_response}")
    print(f"Summary: {remove_think_blocks(summary['output_text'])}")
