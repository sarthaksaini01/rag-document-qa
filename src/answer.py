
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
import ollama

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"
LLM_NAME = "llama3.2:3b"


def main():
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name="my_documents",
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        print("Database empty hai. Pehle python src/embed.py run karo.")
        return

    question = input("\nApna PDF-related question likho: ").strip()

    if not question:
        print("Question khaali nahi hona chahiye.")
        return

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, collection.count())
    )

    documents = results["documents"][0]

    context = "\n\n".join(
        f"Document excerpt {i}:\n{doc}"
        for i, doc in enumerate(documents, start=1)
    )

    prompt = f"""
You are a helpful document question-answering assistant.
Answer the user's question using only the context below.
If the answer is not present in the context, clearly say:
"I could not find this information in the provided document."
Do not invent facts.

Context:
{context}

User question:
{question}

Answer:
"""

    print("\nGenerating answer...\n")

    response = ollama.chat(
        model=LLM_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print("Answer:\n")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
