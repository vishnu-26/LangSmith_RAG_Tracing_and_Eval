# Traceable PDF RAG Application

A command-line Retrieval-Augmented Generation (RAG) application that indexes a PDF into Chroma, retrieves relevant chunks, and answers questions through OpenRouter. LangSmith records a named trace for each important RAG operation.

## Trace structure

```text
Ingest knowledge base
|- Load PDF
|- Split PDF into chunks
`- Embed and index chunks

Answer RAG question
|- Retrieve relevant chunks
|- Format retrieved context
|- Build RAG prompt
|- Call OpenRouter model
`- Parse model response
```

## Prerequisites

- Python 3.13 or later
- An OpenRouter API key
- A LangSmith API key
- A PDF file to use as the knowledge base

## Setup

Install the project dependencies with [uv](https://docs.astral.sh/uv/):

```powershell
uv sync
```

Create a `.env` file in this directory:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
MODEL=nvidia/nemotron-3-ultra-550b-a55b:free

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=kyc-rag

PDF_PATH=C:/path/to/your/document.pdf
CHROMA_PERSIST_DIRECTORY=./chroma_db
RETRIEVER_K=3
```

If your LangSmith workspace is outside the default US region, also set `LANGSMITH_ENDPOINT` to the endpoint for that region.

## Run

```powershell
uv run main.py
```

Ask questions in the terminal. Enter `exit` to close the application.

```text
You: What is the customer onboarding process?
AI: The customer onboarding process is ...
```

The first run loads the configured PDF and creates the Chroma vector store in `chroma_db` (or the directory configured by `CHROMA_PERSIST_DIRECTORY`).

## LangSmith tracing

Tracing is controlled by these environment variables:

| Variable | Purpose |
| --- | --- |
| `LANGSMITH_TRACING` | Enables trace submission; set it to `true`. |
| `LANGSMITH_API_KEY` | Authenticates the application with LangSmith. |
| `LANGSMITH_PROJECT` | Groups traces in a LangSmith project. |
| `LANGSMITH_ENDPOINT` | Optional regional or self-hosted LangSmith endpoint. |

`main.py` uses `@traceable` to distinguish the following operations:

| Operation | Trace name | Run type |
| --- | --- | --- |
| Load the document | `Load PDF` | `chain` |
| Split content | `Split PDF into chunks` | `chain` |
| Create embeddings and index chunks | `Embed and index chunks` | `chain` |
| Retrieve context | `Retrieve relevant chunks` | `retriever` |
| Build model context | `Format retrieved context` | `chain` |
| Render the prompt | `Build RAG prompt` | `prompt` |
| Invoke the chat model | `Call OpenRouter model` | `llm` |
| Convert output to text | `Parse model response` | `parser` |

Open the configured LangSmith project to inspect each operation's inputs, outputs, latency, nested runs, and errors.

## Security

- Never commit `.env`, API keys, or PDF files containing sensitive content.
- Traces can contain prompts and retrieved PDF chunks. Enable LangSmith only for data approved for your LangSmith workspace.
- Rotate an API key immediately if it has been committed or shared unintentionally.

## References

- [LangSmith tracing for LangChain applications](https://docs.langchain.com/langsmith/trace-with-langchain)
- [LangSmith documentation](https://docs.smith.langchain.com/)
