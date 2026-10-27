
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import streamlit

print("Packages imported successfully!")

model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode("Hello RAG")

print("Embedding size:", len(vec))
print("All good!")
