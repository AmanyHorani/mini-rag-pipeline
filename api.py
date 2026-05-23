from fastapi import FastAPI
from pydantic import BaseModel

from pipeline import (
    load_documents,
    chunk_documents,
    build_index,
    retrieve,
    generate_answers
)

app = FastAPI()


# =========================================================
# REQUEST MODEL
# =========================================================

class QuestionRequest(BaseModel):
    question: str


# =========================================================
# LOAD PIPELINE
# =========================================================

documents = load_documents()

chunks = chunk_documents(documents)

vectorizer, vectors = build_index(chunks)


# =========================================================
# API ENDPOINT
# =========================================================

@app.post("/answer")
def answer_question(payload: QuestionRequest):

    query = [{
        "query_id": "API_QUERY",
        "question": payload.question
    }]

    retrieval_results = retrieve(
        query,
        chunks,
        vectorizer,
        vectors
    )

    answers = generate_answers(
        retrieval_results
    )

    return answers[0]