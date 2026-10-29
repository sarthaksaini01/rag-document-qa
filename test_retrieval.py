
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# MySQL Handbook ke liye test questions
test_questions = [
    {
        "question": "What is a primary key in MySQL?",
        "expected_keywords": ["primary key", "uniquely identifies"],
    },
    {
        "question": "What is a foreign key in MySQL?",
        "expected_keywords": ["foreign key", "two tables"],
    },
    {
        "question": "What is a database in MySQL?",
        "expected_keywords": ["database", "related data"],
    },
]

# Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Connect to existing ChromaDB
client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_collection(name="pdf_documents")

if collection.count() == 0:
    print("Database empty hai. Pehle app mein PDF index karo.")
    raise SystemExit(1)

print(f"Total indexed chunks: {collection.count()}")
print("\nStarting retrieval evaluation...\n")

passed_tests = 0

for index, test in enumerate(test_questions, start=1):
    question = test["question"]
    expected_keywords = test["expected_keywords"]

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    # Same document + page + content ko repeat nahi karna
    unique_results = []
    seen = set()

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        key = (
            metadata.get("source", ""),
            metadata.get("page", ""),
            document
        )

        if key not in seen:
            seen.add(key)
            unique_results.append((document, metadata, distance))

    # Check whether expected keywords appear in retrieved content
    combined_text = " ".join(
        document.lower()
        for document, _, _ in unique_results
    )

    found_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() in combined_text
    ]

    keyword_found = len(found_keywords) > 0

    if keyword_found:
        passed_tests += 1

    print(f"Test {index}: {question}")
    print(f"Expected keywords: {expected_keywords}")
    print(f"Found keywords: {found_keywords}")
    print(f"Keyword check: {'PASS' if keyword_found else 'CHECK'}")

    if unique_results:
        best_distance = unique_results[0][2]
        print(f"Best cosine distance: {best_distance:.3f}")

    for document, metadata, distance in unique_results:
        print(
            f"  Source: {metadata.get('source', 'Unknown')} | "
            f"Page: {metadata.get('page', 'Unknown')} | "
            f"Cosine distance: {distance:.3f}"
        )
        print(f"  Preview: {document[:180]}...")
    
    print("-" * 60)

print("\nEvaluation complete!")
print(f"Keyword checks passed: {passed_tests}/{len(test_questions)}")
print(
    "Note: PASS means an expected keyword was found; "
    "it does not guarantee a correct answer."
)
