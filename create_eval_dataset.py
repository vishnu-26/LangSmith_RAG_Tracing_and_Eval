import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "evaluation_data" / "kyc_rag_eval_samples.jsonl"
DATASET_NAME = "kyc-rag-evaluation"

load_dotenv(PROJECT_DIR / ".env")

client = Client()

if client.has_dataset(dataset_name=DATASET_NAME):
    raise SystemExit(
        f"Dataset '{DATASET_NAME}' already exists. "
        "Choose a new name or delete the existing dataset first."
    )

with DATASET_PATH.open(encoding="utf-8") as file:
    rows = [json.loads(line) for line in file if line.strip()]

# Keep the question/reference answer format and retain source-page data as metadata.
examples = [
    {
        "inputs": row["inputs"],
        "outputs": row["outputs"],
        "metadata": {
            **row["metadata"],
            "sample_id": row["id"],
        },
    }
    for row in rows
]

dataset = client.create_dataset(
    DATASET_NAME,
    description=(
        "Grounded question-answer examples for evaluating the "
        "Oracle Financial Services KYC PDF RAG application."
    ),
)

client.create_examples(
    dataset_id=dataset.id,
    examples=examples,
)

print(f"Uploaded {len(examples)} examples to '{DATASET_NAME}'.")