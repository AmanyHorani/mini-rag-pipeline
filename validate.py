import json
import os

REQUIRED_FILES = [
    "artifacts/chunks.json",
    "artifacts/retrieval.json",
    "artifacts/answers.json",
    "artifacts/eval.json"
]

ALLOWED_ANSWER_LABELS = {
    "grounded_answer",
    "insufficient_context",
    "conflicting_context"
}

ALLOWED_RETRIEVAL_STATUSES = {
    "hit",
    "partial_hit",
    "miss"
}

def validate():

    for file in REQUIRED_FILES:

        if not os.path.exists(file):
            raise Exception(f"Missing artifact: {file}")

    with open("artifacts/retrieval.json", "r") as f:
        retrieval = json.load(f)

    with open("artifacts/answers.json", "r") as f:
        answers = json.load(f)

    with open("artifacts/eval.json", "r") as f:
        evaluation = json.load(f)

    for item in retrieval:

        assert len(item["top_k"]) >= 3

        for chunk in item["top_k"]:
            assert isinstance(chunk["score"], (int, float))

    for answer in answers:

        assert answer["answer_label"] in ALLOWED_ANSWER_LABELS

        if answer["answer_label"] == "grounded_answer":
            assert len(answer["citations"]) > 0

    for item in evaluation["evaluations"]:

        assert item["retrieval_status"] in ALLOWED_RETRIEVAL_STATUSES

    assert "summary" in evaluation

    print("Validation successful!")

if __name__ == "__main__":
    validate()