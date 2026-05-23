import os
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONTROLLED VOCABULARIES
# =========================================================

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

# =========================================================
# PIPELINE STAGES
# =========================================================

PIPELINE_STAGES = [
    "INIT",
    "DOCUMENTS_LOADED",
    "DOCUMENTS_CHUNKED",
    "INDEX_BUILT",
    "RETRIEVAL_COMPLETE",
    "ANSWERS_GENERATED",
    "EVALUATION_COMPLETE",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED"
]

current_stage = "INIT"

# =========================================================
# PATHS
# =========================================================

KB_DIR = "kb"
ARTIFACTS_DIR = "artifacts"

Path(ARTIFACTS_DIR).mkdir(exist_ok=True)

# =========================================================
# DOCUMENT INGESTION
# =========================================================

def load_documents():
    global current_stage

    documents = []

    for file_name in os.listdir(KB_DIR):

        file_path = os.path.join(KB_DIR, file_name)

        if not file_name.endswith(".txt"):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        title_match = re.search(r"Title:\s*(.*)", content)
        section_match = re.search(r"Section:\s*(.*)", content)

        title = title_match.group(1).strip()
        section = section_match.group(1).strip()

        body = re.split(r"Section:.*\n", content, maxsplit=1)[1].strip()

        documents.append({
            "title": title,
            "section": section,
            "body": body
        })

    current_stage = "DOCUMENTS_LOADED"

    return documents

# =========================================================
# CHUNKING
# =========================================================

def chunk_documents(documents, chunk_size=220):

    global current_stage

    chunks = []

    chunk_counter = 1

    for doc in documents:

        text = doc["body"]

        start = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end]

            chunk = {
                "chunk_id": f"chunk_{chunk_counter}",
                "doc_title": doc["title"],
                "section": doc["section"],
                "text": chunk_text,
                "start_char": start,
                "end_char": end
            }

            chunks.append(chunk)

            chunk_counter += 1

            start = end

    with open(f"{ARTIFACTS_DIR}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    current_stage = "DOCUMENTS_CHUNKED"

    return chunks

# =========================================================
# INDEX BUILDING
# =========================================================

def build_index(chunks):

    global current_stage

    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform(texts)

    current_stage = "INDEX_BUILT"

    return vectorizer, vectors

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(queries, chunks, vectorizer, vectors, top_k=3):

    global current_stage

    retrieval_results = []

    chunk_texts = [chunk["text"] for chunk in chunks]

    for query in queries:

        question = query["question"]

        query_vector = vectorizer.transform([question])

        similarities = cosine_similarity(query_vector, vectors).flatten()

        ranked_indices = similarities.argsort()[::-1]

        top_chunks = []

        for rank, idx in enumerate(ranked_indices[:top_k], start=1):

            chunk = chunks[idx]

            top_chunks.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "doc_title": chunk["doc_title"],
                "score": float(similarities[idx]),
                "chunk_text": chunk["text"]
            })

        retrieval_results.append({
            "query_id": query["query_id"],
            "question": question,
            "top_k": top_chunks
        })

    with open(f"{ARTIFACTS_DIR}/retrieval.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, indent=2)

    current_stage = "RETRIEVAL_COMPLETE"

    return retrieval_results

# =========================================================
# ANSWER GENERATION
# =========================================================

def generate_answers(retrieval_results):

    global current_stage

    answers = []

    for result in retrieval_results:

        best_chunk = result["top_k"][0]

        citation = f"[{best_chunk['doc_title']} §{best_chunk['chunk_id']}]"

        score = best_chunk["score"]

        if score < 0.05:

            answer_label = "insufficient_context"

            answer = "Insufficient context to answer the question."

            citations = []

            used_chunk_ids = []

        else:

            answer_label = "grounded_answer"

            answer = f"{best_chunk['chunk_text']} {citation}"

            citations = [citation]

            used_chunk_ids = [best_chunk["chunk_id"]]

        answers.append({
            "query_id": result["query_id"],
            "answer_label": answer_label,
            "answer": answer,
            "citations": citations,
            "used_chunk_ids": used_chunk_ids
        })

    with open(f"{ARTIFACTS_DIR}/answers.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2)

    current_stage = "ANSWERS_GENERATED"

    return answers

# =========================================================
# EVALUATION
# =========================================================

def evaluate(queries, retrieval_results):

    global current_stage

    evaluations = []

    hits = 0
    partial_hits = 0
    misses = 0

    for query, retrieval in zip(queries, retrieval_results):

        expected_titles = query["expected_doc_titles"]

        retrieved_titles = [
            item["doc_title"]
            for item in retrieval["top_k"]
        ]

        matched = any(
            title in retrieved_titles
            for title in expected_titles
        )

        if matched:

            retrieval_status = "hit"

            explanation = "Expected title found in top 3"

            hits += 1

        else:

            retrieval_status = "miss"

            explanation = "Expected title not found"

            misses += 1

        evaluations.append({
            "query_id": query["query_id"],
            "expected_doc_titles": expected_titles,
            "retrieved_doc_titles_top3": retrieved_titles,
            "retrieval_status": retrieval_status,
            "matched_expected_title": matched,
            "explanation": explanation
        })

    summary = {
        "top3_hit_rate": hits / len(queries),
        "total_queries": len(queries),
        "hits": hits,
        "partial_hits": partial_hits,
        "misses": misses
    }

    eval_output = {
        "evaluations": evaluations,
        "summary": summary
    }

    with open(f"{ARTIFACTS_DIR}/eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)

    current_stage = "EVALUATION_COMPLETE"

    return eval_output

# =========================================================
# GROUNDING CHECK
# =========================================================

def grounding_check(answers, retrieval_results):

    checks = []

    retrieval_map = {}

    for retrieval in retrieval_results:

        for chunk in retrieval["top_k"]:

            retrieval_map[chunk["chunk_id"]] = chunk["chunk_text"]

    for answer in answers:

        valid = True

        explanation = "All citations valid"

        for chunk_id in answer["used_chunk_ids"]:

            if chunk_id not in retrieval_map:

                valid = False

                explanation = "Citation refers to non-retrieved chunk"

        checks.append({
            "query_id": answer["query_id"],
            "grounding_valid": valid,
            "explanation": explanation
        })

    with open(f"{ARTIFACTS_DIR}/grounding_check.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

# =========================================================
# CHUNKING COMPARISON
# =========================================================

def chunking_comparison():

    comparison = {
        "strategies": [
            {
                "name": "fixed_size",
                "description": "Deterministic character-based chunking"
            },
            {
                "name": "paragraph_based",
                "description": "Better semantic preservation but variable chunk sizes"
            }
        ],
        "tradeoff": (
            "Fixed-size chunking is simpler and deterministic, "
            "while paragraph chunking preserves semantic coherence better."
        )
    }

    with open(f"{ARTIFACTS_DIR}/chunking_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

# =========================================================
# MAIN
# =========================================================

def main():

    print("Starting pipeline...")

    documents = load_documents()

    chunks = chunk_documents(documents)

    vectorizer, vectors = build_index(chunks)

    with open("queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    retrieval_results = retrieve(
        queries,
        chunks,
        vectorizer,
        vectors
    )

    answers = generate_answers(retrieval_results)

    evaluate(queries, retrieval_results)

    grounding_check(answers, retrieval_results)

    chunking_comparison()

    global current_stage

    current_stage = "RESULTS_FINALISED"

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()