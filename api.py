import json

from fastapi import FastAPI

from pipeline import (
    load_documents,
    chunk_documents,
    build_index,
    retrieve,
    generate_answers
)

app = FastAPI()

documents = load_documents()

chunks = chunk_documents(documents)

vectorizer, vectors = build_index(chunks)

@app.post("/answer")
def answer_question(payload: dict):

    question = payload["question"]

    query = [{
        "query_id": "API_QUERY",
        "question": question
    }]

    retrieval_results = retrieve(
        query,
        chunks,
        vectorizer,
        vectors
    )

    answers = generate_answers(retrieval_results)

    return answers[0]