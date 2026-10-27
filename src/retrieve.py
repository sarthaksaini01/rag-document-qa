
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name="my_documents",
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        print("Database empty hai. Pehle python src/embed.py run karo.")
        return

    question = input("\nApna question likho: ").strip()

    if not question:
        print("Question khaali nahi hona chahiye.")
        return

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, collection.count())
    )

    print("\n--- Relevant Document Chunks ---\n")

    for i, document in enumerate(results["documents"][0], start=1):
        metadata = results["metadatas"][0][i - 1]
        print(f"Result {i}")
        print(f"Source: {metadata['source']}")
        print(f"Chunk: {metadata['chunk_index']}")
        print(document)
        print("-" * 50)


if __name__ == "__main__":
    main()
