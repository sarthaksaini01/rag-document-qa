
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from ingest import load_pdf, split_text


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("data folder mein koi PDF nahi mili.")
        return

    # Embedding model load karo
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Local vector database
    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name="my_documents",
        metadata={"hnsw:space": "cosine"}
    )

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        text = load_pdf(pdf_path)
        chunks = split_text(text, chunk_size=500, overlap=50)

        if not chunks:
            print("Is PDF mein extractable text nahi mila.")
            continue

        embeddings = model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=True
        ).tolist()

        # Har chunk ki unique ID aur source metadata
        ids = [
            f"{pdf_path.name}_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": pdf_path.name,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]

        # Re-run par same IDs overwrite honge,
        # duplicate chunks add nahi honge.
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Stored {len(chunks)} chunks from {pdf_path.name}")

    print("\nTotal chunks in database:", collection.count())
    print("Embedding and storage complete!")


if __name__ == "__main__":
    main()
