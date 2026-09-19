import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load the .env stored beside this script.
APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

PDF_PATH = os.getenv(
    "PDF_PATH",
    "C:/Users/Vishnu/Documents/FCCM/OFSAA_BEHAVIOR_DETECTION_8.X/"
    "BEHAVIOR_DETECTION_8.1/branches/8.1.2.7/KYC_Guides/"
    "oracle-financial-services-know-your-customer-administration-guide.pdf",
)
CHROMA_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY",
    str(APP_DIR / "chroma_db"),
)
TOP_K = int(os.getenv("RETRIEVER_K", "3"))

augmentation_prompt = PromptTemplate(
    template="""
You are an AI assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
""".strip(),
    input_variables=["context", "question"],
)

embedding_model = None
llm = None
vectorstore = None
retriever = None


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def initialize_models() -> None:
    """Keep credentials out of trace inputs."""
    global embedding_model, llm

    api_key = require_env("OPENROUTER_API_KEY")

    embedding_model = NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b:free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    llm = ChatOpenRouter(
        model=require_env("MODEL"),
        api_key=api_key,
    )


@traceable(name="Load PDF", run_type="chain", tags=["rag", "ingestion", "pdf-loader"],)
def load_pdf(pdf_path: str) -> list[Document]:
    return PyPDFLoader(pdf_path).load()


@traceable(name="Split PDF into chunks", run_type="chain", tags=["rag", "ingestion", "text-splitting"])
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


@traceable(name="Embed and index chunks", run_type="chain", tags=["rag", "ingestion", "vector-store"],)
def index_documents(chunks: list[Document]) -> dict[str, int | str]:
    global vectorstore, retriever

    if embedding_model is None:
        raise RuntimeError("Models have not been initialized.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIRECTORY,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    return {
        "chunks_indexed": len(chunks),
        "top_k": TOP_K,
        "persist_directory": CHROMA_DIRECTORY,
    }


@traceable(name="Ingest knowledge base", run_type="chain", tags=["rag", "ingestion"],)
def ingest_pdf(pdf_path: str) -> dict[str, int | str]:
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    result = index_documents(chunks)
    result["pages_loaded"] = len(documents)
    return result


@traceable(name="Retrieve relevant chunks", run_type="retriever", tags=["rag", "retrieval"],)
def retrieve_documents(question: str) -> list[Document]:
    if retriever is None:
        raise RuntimeError("Knowledge base has not been indexed.")

    return retriever.invoke(question)


@traceable(name="Format retrieved context", run_type="chain", tags=["rag", "context"],)
def format_context(documents: list[Document]) -> str:
    return "\n\n---\n\n".join(
        document.page_content for document in documents
    )


@traceable(name="Build RAG prompt", run_type="prompt", tags=["rag", "prompt"],)
def build_prompt(question: str, context: str) -> str:
    return augmentation_prompt.format(
        question=question,
        context=context,
    )


@traceable(name="Call OpenRouter model", run_type="llm", tags=["rag", "generation", "openrouter"],)
def generate_response(prompt: str):
    if llm is None:
        raise RuntimeError("Models have not been initialized.")

    return llm.invoke(prompt)


@traceable(name="Parse model response", run_type="parser", tags=["rag", "output-parsing"],)
def parse_response(response) -> str:
    return StrOutputParser().invoke(response)


@traceable(name="Answer RAG question", run_type="chain", tags=["rag", "question-answering"],)
def answer_question(question: str) -> str:
    documents = retrieve_documents(question)
    context = format_context(documents)
    prompt = build_prompt(question, context)
    response = generate_response(prompt)
    return parse_response(response)


def main() -> None:
    initialize_models()

    ingestion_summary = ingest_pdf(PDF_PATH)
    print(
        f"Indexed {ingestion_summary['chunks_indexed']} chunks "
        f"from {ingestion_summary['pages_loaded']} PDF pages."
    )

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            break

        if not user_input:
            continue

        answer = answer_question(user_input)
        print(f"AI: {answer}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Flush background trace submissions before the program exits.
        wait_for_all_tracers()