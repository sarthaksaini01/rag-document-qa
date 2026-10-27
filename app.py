
from pathlib import Path
import hashlib
import time
import chromadb
import ollama
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:1.5b"

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚")
st.title("📚 RAG Document Q&A")
st.write("PDF upload karo aur uske content par questions poochho.")

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks

@st.cache_resource
def load_resources():
    start_time = time.time()
    print("Loading embedding model...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name="pdf_documents",
        metadata={"hnsw:space": "cosine"}
    )

    print(f"ChromaDB connected in {time.time() - start_time:.2f} seconds")

    return embedding_model, collection


start_time = time.time()
embedding_model, collection = load_resources()
print(f"Total resource loading time: {time.time() - start_time:.2f} seconds")




def index_pdf(filename, pdf_bytes):
    import io

    # Full SHA-256 hash: same file ko identify karne ke liye
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Check whether this exact PDF has already been indexed
    existing = collection.get(
        where={"file_hash": pdf_hash},
        include=["metadatas"],
        limit=1
    )

    if existing["ids"]:
        return {
            "status": "duplicate",
            "chunks": 0,
            "message": f"{filename} pehle se indexed hai."
        }

    reader = PdfReader(io.BytesIO(pdf_bytes))

    all_chunks = []
    all_metadatas = []
    pages_with_text = 0

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()

        if not page_text:
            continue

        pages_with_text += 1
        chunks = split_text(page_text)

        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            all_metadatas.append({
                "source": filename,
                "page": page_number,
                "chunk_index": chunk_index,
                "file_hash": pdf_hash
            })

    if not all_chunks:
        raise ValueError(
            "PDF se readable text nahi mila. "
            "Scanned PDF ke liye OCR ki zaroorat ho sakti hai."
        )

    ids = [
        f"{pdf_hash}_{i}"
        for i in range(len(all_chunks))
    ]

    embeddings = embedding_model.encode(
        all_chunks,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas
    )

    print(
        f"{filename}: {pages_with_text}/{len(reader.pages)} pages "
        "mein readable text mila."
    )

    return {
        "status": "indexed",
        "chunks": len(all_chunks),
        "message": f"{filename} successfully indexed."
    }


with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_files = st.file_uploader(
        "PDF files select karo",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Index uploaded PDFs"):
        if not uploaded_files:
            st.warning("Pehle kam se kam ek PDF upload karo.")
        else:
            total = 0

            for file in uploaded_files:
                try:
                    result = index_pdf(file.name, file.getvalue())

                    if result["status"] == "duplicate":
                        st.info(result["message"])
                    else:
                        total += result["chunks"]
                        st.success(
                            f'{result["message"]} '
                            f'Chunks: {result["chunks"]}'
                        )

                except Exception as e:
                    st.error(f"{file.name}: {e}")

            st.info(f"Is attempt mein naye chunks indexed: {total}")

if collection.count() == 0:
    st.info("Shuru karne ke liye sidebar se PDF upload aur index karo.")
    st.stop()

st.caption(f"Database mein total chunks: {collection.count()}")


if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Apni PDF ke baare mein question poochho...")


if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Document mein answer search ho raha hai..."):
            try:
                # 1. Question embedding ka time
                start_time = time.time()

                query_embedding = embedding_model.encode(
                    question,
                    normalize_embeddings=True
                ).tolist()

                print(
                    f"Question embedding time: "
                    f"{time.time() - start_time:.2f} seconds"
                )

                # 2. Document search ka time
                start_time = time.time()

                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(3, collection.count())
                )

                print(
                    f"Document search time: "
                    f"{time.time() - start_time:.2f} seconds"
                )


                docs = results["documents"][0]
                metadatas = results["metadatas"][0]

                context = "\n\n".join(
                    f"[Source: {metadata.get('source', 'Unknown')}, "
                    f"Page: {metadata.get('page', 'Unknown')}]\n{doc}"
                    for doc, metadata in zip(docs, metadatas)
                )

                chat_history = st.session_state.messages[:-1][-6:]
                history_text = "\n".join(
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: "
                    f"{msg['content']}"
                    for msg in chat_history
                )

                prompt = f"""
You are a helpful document question-answering assistant.
Answer using the document context provided below.
If the answer is not in the context, say that you could not find it in the document.
Keep the answer clear and simple.

Recent chat history:
{history_text}

Document context:
{context}

Current question:
{question}

Answer:
"""

                
                start_time = time.time()

                response = ollama.chat(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    options={
                        "num_predict": 150,
                        "temperature": 0.2
                    },
                    keep_alive="10m"
                )

                print(
                    f"Ollama answer generation time: "
                    f"{time.time() - start_time:.2f} seconds"
                )

                answer = response["message"]["content"]
                st.markdown(answer)

                st.markdown("**Sources:**")
                for doc, metadata in zip(docs, metadatas):
                    source = metadata.get("source", "Unknown")
                    page = metadata.get("page", "Unknown")

                    with st.expander(f"📄 {source} — Page {page}"):
                        st.write(doc)

            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
