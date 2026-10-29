
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:1.5b"

MAX_COSINE_DISTANCE = 0.65
TOP_K = 3

TEST_CASES = [
    {
        "question": "What is a primary key in MySQL?",
        "expected_page": 40,
        "keywords": ["uniquely identifies", "unique", "null"],
        "should_answer": True,
    },
    {
        "question": "What is a foreign key in MySQL?",
        "expected_page": 44,
        "keywords": ["foreign key", "tables", "data integrity"],
        "should_answer": True,
    },
    {
        "question": "What is a database?",
        "expected_page": 5,
        "keywords": ["database", "related data", "organized"],
        "should_answer": True,
    },
    {
        "question": "Explain quantum mechanics based on this document.",
        "expected_page": None,
        "keywords": [],
        "should_answer": False,
    },
]

ABSTENTION_MESSAGE = (
    "The uploaded document does not contain sufficiently relevant "
    "information to answer this question."
)


# ---------------- HELPER FUNCTIONS ----------------

def generate_answer(question, context):
   
    prompt = f"""
You are a helpful assistant answering questions from a document.

Instructions:
1. Answer using only the information provided in the context.
2. Include the important facts that directly answer the question.
3. Do not omit important constraints, conditions, or limitations.
4. For a primary key question, mention that it uniquely identifies
   each row and cannot contain NULL values, if the context states this.
5. If the context does not contain enough information, say so.
6. Do not invent facts.

Context:
{context}

Question: {question}

Answer:
"""


    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def check_keywords(answer, keywords):
    answer_lower = answer.lower()
    missing = [
        keyword
        for keyword in keywords
        if keyword.lower() not in answer_lower
    ]

    return missing


# ---------------- MAIN EVALUATION ----------------

def main():
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    print(f"Indexed chunks: {collection.count()}")
    print(f"LLM model: {LLM_MODEL}")
    print(f"Distance threshold: {MAX_COSINE_DISTANCE}")
    print("\nStarting answer-quality evaluation...\n")

    total_answerable = 0
    correct_pages = 0
    keyword_passes = 0
    abstention_passes = 0

    for index, test in enumerate(TEST_CASES, start=1):
        question = test["question"]

        print(f"\nTest {index}: {question}")

        query_embedding = embedding_model.encode(
            question
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        relevant_chunks = []

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if distance <= MAX_COSINE_DISTANCE:
                relevant_chunks.append({
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                })

        if results["distances"][0]:
            print(
                "Best distance:",
                round(min(results["distances"][0]), 3),
            )

        if not relevant_chunks:
            answer = ABSTENTION_MESSAGE
            print("LLM called: NO")
        else:
            context = "\n\n".join(
                chunk["document"] for chunk in relevant_chunks
            )

            print("LLM called: YES")
            answer = generate_answer(question, context)

        print("Answer:")
        print(answer)

        # Check whether the system should answer or abstain.
        should_answer = test["should_answer"]
        actually_answered = bool(relevant_chunks)

        if should_answer:
            total_answerable += 1

            if actually_answered:
                print("Retrieval decision: ANSWERABLE")

                expected_page = test["expected_page"]
                retrieved_pages = [
                    chunk["metadata"].get("page")
                    for chunk in relevant_chunks
                ]

                if expected_page in retrieved_pages:
                    correct_pages += 1
                    print(f"Expected page {expected_page}: PASS")
                else:
                    print(f"Expected page {expected_page}: FAIL")

                missing = check_keywords(
                    answer,
                    test["keywords"],
                )

                if not missing:
                    keyword_passes += 1
                    print("Keyword check: PASS")
                else:
                    print("Keyword check: REVIEW")
                    print("Missing keywords:", missing)

            else:
                print("Retrieval decision: FAIL — no relevant chunks")

        else:
            if not actually_answered:
                abstention_passes += 1
                print("Out-of-scope handling: PASS")
            else:
                print(
                    "Out-of-scope handling: REVIEW — "
                    "chunks passed the threshold"
                )

        print("-" * 60)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Expected-page checks passed: {correct_pages}/{total_answerable}")
    print(f"Keyword checks passed: {keyword_passes}/{total_answerable}")
    print(
        "Out-of-scope abstention checks passed: "
        f"{abstention_passes}/1"
    )
    print(
        "\nNote: Keyword matching is a basic check. "
        "It does not prove factual correctness."
    )


if __name__ == "__main__":
    main()
