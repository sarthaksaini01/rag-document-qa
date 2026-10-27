# 📚 RAG-Based Document Question Answering System

A Python-based application that allows users to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG).

## 🚀 Overview

This project combines semantic search with a Large Language Model (LLM) to retrieve relevant information from PDF documents and generate context-aware answers.

Instead of relying only on the model's general knowledge, the application searches the uploaded document for relevant text and uses that information to answer questions.

## ✨ Features

- Upload PDF documents through a Streamlit interface.
- Extract text from PDFs using PyPDF.
- Split document text into overlapping chunks.
- Generate text embeddings using Sentence Transformers.
- Store document chunks and embeddings in ChromaDB.
- Retrieve relevant document sections using semantic similarity.
- Generate answers using a local language model through Ollama.
- Display source document and page information.
- Avoid indexing identical PDFs more than once.

## 📸 Application Demo

![RAG Document Q&A Application](screenshots/rag-app-demo.png)

## 🛠️ Technologies Used

- **Language:** Python
- **Frontend:** Streamlit
- **PDF Processing:** PyPDF
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **Language Model Runtime:** Ollama
- **LLM:** Qwen 2.5 1.5B

## ⚙️ How It Works

1. The user uploads a PDF document.
2. The application extracts readable text from the PDF.
3. The extracted text is divided into smaller, overlapping chunks.
4. Sentence Transformers converts the chunks into vector embeddings.
5. ChromaDB stores the chunks, embeddings, and document metadata.
6. When a user asks a question, the application creates an embedding for the query.
7. ChromaDB retrieves the most relevant document chunks.
8. Ollama generates an answer using the retrieved context.
9. The application displays the answer and source-page information.

## 📋 Prerequisites

Before running the project, install:

- Python 3.10 or a compatible Python version supported by the dependencies.
- Git.
- Ollama.

You will also need sufficient memory and disk space to run the embedding model and local language model.

## 🔧 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sarthaksaini01/rag-document-qa.git
cd rag-document-qa
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama from its official website:

https://ollama.com/

Download the model used by the application:

```bash
ollama pull qwen2.5:1.5b
```

Make sure Ollama is running before starting the application.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

`http://localhost:8501`

## 📖 Usage

1. Open the application in your browser.
2. Upload one or more PDF documents using the sidebar.
3. Click **Index uploaded PDFs**.
4. Enter a question related to the indexed documents.
5. Read the generated answer and inspect the source-page references.

## 📁 Project Structure

```text
rag-document-qa/
├── app.py
├── requirements.txt
├── test_setup.py
├── src/
└── .gitignore
```

The exact contents of the `src/` directory depend on the project files included in the repository.

## 🔍 Key Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Text chunking and overlap
- Semantic search
- Text embeddings
- Vector databases
- Large Language Model integration
- PDF processing
- Interactive application development with Streamlit

## ⚠️ Limitations

- Answer quality depends on the text extracted from the PDF and the relevance of retrieved chunks.
- Scanned PDFs may require OCR before their text can be searched.
- Local model performance depends on available system resources.
- The application is designed to answer from retrieved document context; answers should be checked against the cited source pages.

## 👨‍💻 Author

**Sarthak Saini**

GitHub: https://github.com/sarthaksaini01

---

⭐ If you find this project useful, feel free to explore the code and suggest improvements.
