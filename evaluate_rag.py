from __future__ import annotations

import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client
from langsmith.evaluation import run_evaluator

import main as rag


DATASET_NAME = "kyc-rag-evaluation"

load_dotenv(rag.APP_DIR / ".env")
client = Client()


def rag_target(inputs: dict[str, str]) -> dict[str, Any]:
    """Run the same RAG steps as the application and keep retrieval evidence."""
    question = inputs["question"]

    documents = rag.retrieve_documents(question)
    context = rag.format_context(documents)
    prompt = rag.build_prompt(question, context)
    response = rag.generate_response(prompt)
    answer = rag.parse_response(response)

    retrieved_documents = [
        {
            "page": document.metadata.get("page"),
            "source": document.metadata.get("source"),
            "content": document.page_content,
        }
        for document in documents
    ]

    return {
        "answer": answer,
        "retrieved_documents": retrieved_documents,
    }


def retrieved_context(run) -> str:
    """Create readable evidence for the groundedness-related evaluators."""
    documents = (run.outputs or {}).get("retrieved_documents", [])
    return "\n\n---\n\n".join(
        f"[Source page: {document.get('page')}]\n{document.get('content', '')}"
        for document in documents
    )


def judge_metric(metric: str, criterion: str, evidence: str) -> dict[str, Any]:
    """Ask the evaluator model for one metric score and a brief explanation."""
    prompt = f"""
You are evaluating a RAG application.

Criterion: {criterion}

{evidence}

Give a score from 0.0 to 1.0, where 1.0 fully meets the criterion, 0.5
partially meets it, and 0.0 does not meet it.

Return exactly two lines:
SCORE: <number>
REASON: <brief explanation>
""".strip()

    response = str(rag.llm.invoke(prompt).content)
    score_match = re.search(r"SCORE:\s*([01](?:\.\d+)?)", response)
    reason_match = re.search(r"REASON:\s*(.*)", response, re.DOTALL)

    score = float(score_match.group(1)) if score_match else 0.0
    return {
        "key": metric,
        "score": min(score, 1.0),
        "comment": reason_match.group(1).strip() if reason_match else response,
    }


@run_evaluator
def correctness(run, example):
    """Does the RAG answer agree with the dataset's reference answer?"""
    return judge_metric(
        "correctness",
        "The RAG answer is factually consistent with the reference answer.",
        f"""Question: {example.inputs['question']}
Reference answer: {(example.outputs or {})['reference_answer']}
RAG answer: {(run.outputs or {}).get('answer', '')}""",
    )


@run_evaluator
def relevance(run, example):
    """Does the RAG answer directly answer the user's question?"""
    return judge_metric(
        "relevance",
        "The RAG answer directly addresses the question without material off-topic information.",
        f"""Question: {example.inputs['question']}
RAG answer: {(run.outputs or {}).get('answer', '')}""",
    )


@run_evaluator
def groundedness(run, example):
    """Are the answer's claims supported by the retrieved context?"""
    return judge_metric(
        "groundedness",
        "Every material claim in the RAG answer is supported by the retrieved context.",
        f"""Retrieved context:
{retrieved_context(run)}

RAG answer: {(run.outputs or {}).get('answer', '')}""",
    )


@run_evaluator
def retrieval_relevance(run, example):
    """Are the retrieved chunks useful for answering the question?"""
    return judge_metric(
        "retrieval_relevance",
        "The retrieved context contains information useful for answering the question.",
        f"""Question: {example.inputs['question']}

Retrieved context:
{retrieved_context(run)}""",
    )


def main() -> None:
    if not client.has_dataset(dataset_name=DATASET_NAME):
        raise RuntimeError(
            f"Dataset '{DATASET_NAME}' was not found. "
            "Run upload_evaluation_dataset.py first."
        )

    rag.initialize_models()
    rag.ingest_pdf(rag.PDF_PATH)

    results = client.evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[
            correctness,
            relevance,
            groundedness,
            retrieval_relevance,
        ],
        experiment_prefix="kyc-rag-four-metrics",
        description=(
            "RAG quality evaluation with correctness, relevance, groundedness, "
            "and retrieval relevance."
        ),
        max_concurrency=1,
    )

    print(f"Experiment: {results.experiment_name}")
    print(f"Results: {results.url}")


if __name__ == "__main__":
    try:
        main()
    finally:
        wait_for_all_tracers()
