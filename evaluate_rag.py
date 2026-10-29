
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# Test cases for MySQL Handbook.pdf
TEST_CASES = [
    {
        "question": "What is a primary key in MySQL?",
        "expected_keywords": ["primary key", "uniquely identifies"],
        "expected_page": 40,
    },
    {
        "question": "What is a foreign key in MySQL?",
        "expected_keywords": ["foreign key", "two tables"],
        "expected_page": 44,
    },
    {
        "question": "What is a database in MySQL?",
        "expected_keywords": ["database", "related data"],
        "expected_page": 5,
    },
]


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name="pdf_documents")

    if collection.count() == 0:
        print("No indexed chunks found. Please index a PDF first.")
        return

    print(f"Total indexed chunks: {collection.count()}")
    print("\nStarting automated evaluation...\n")

    retrieval_passes = 0
    keyword_passes = 0
    citation_page_passes = 0

    for number, test in enumerate(TEST_CASES, start=1):
        question = test["question"]

        embedding = model.encode(
            question,
            normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(TOP_K, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0] or []
        metadatas = results["metadatas"][0] or []
        distances = results["distances"][0] or []

        # Avoid counting repeated chunks more than once
        unique_results = []
        seen = set()

        for doc, metadata, distance in zip(
            documents, metadatas, distances
        ):
            key = (
                metadata.get("source", ""),
                metadata.get("page", ""),
                doc,
            )

            if key not in seen:
                seen.add(key)
                unique_results.append((doc, metadata, distance))

        retrieved_text = " ".join(
            doc.lower() for doc, _, _ in unique_results
        )

        found_keywords = [
            keyword
            for keyword in test["expected_keywords"]
            if keyword.lower() in retrieved_text
        ]

        keyword_pass = len(found_keywords) == len(
            test["expected_keywords"]
        )

        expected_page_found = any(
            str(metadata.get("page", "")) == str(test["expected_page"])
            for _, metadata, _ in unique_results
        )

        retrieval_pass = bool(unique_results)

        retrieval_passes += int(retrieval_pass)
        keyword_passes += int(keyword_pass)
        citation_page_passes += int(expected_page_found)

        print(f"Test {number}: {question}")
        print(
            f"  Retrieval: "
            f"{'PASS' if retrieval_pass else 'FAIL'}"
        )
        print(
            f"  Expected keywords: "
            f"{'PASS' if keyword_pass else 'CHECK'}"
        )
        print(f"  Keywords found: {found_keywords}")
        print(
            f"  Expected page {test['expected_page']} retrieved: "
            f"{'PASS' if expected_page_found else 'CHECK'}"
        )

        if unique_results:
            best_doc, best_metadata, best_distance = unique_results[0]
            print(
                f"  Best result: {best_metadata.get('source', 'Unknown')} "
                f"| Page {best_metadata.get('page', 'Unknown')} "
                f"| Distance {best_distance:.3f}"
            )

        print("-" * 55)

    total = len(TEST_CASES)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Retrieval checks: {retrieval_passes}/{total}")
    print(f"Keyword checks: {keyword_passes}/{total}")
    print(f"Expected-page checks: {citation_page_passes}/{total}")
    print("\nEvaluation complete.")
    print(
        "Note: This tests retrieval and expected pages, "
        "not the correctness of generated answers."
    )


if __name__ == "__main__":
    main()
