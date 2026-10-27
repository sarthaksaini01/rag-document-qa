
from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path):
    """PDF se text extract karta hai."""
    reader = PdfReader(pdf_path)
    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages_text.append(text)

    return "\n".join(pages_text)


def split_text(text, chunk_size=500, overlap=50):
    """Text ko overlapping chunks mein divide karta hai."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

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


if __name__ == "__main__":
    data_folder = Path(__file__).resolve().parent.parent / "data"
    pdf_files = list(data_folder.glob("*.pdf"))

    if not pdf_files:
        print("data folder mein koi PDF nahi mili.")
        print("Apni PDF ko rag-qa-tool/data/ mein rakho.")
    else:
        pdf_path = pdf_files[0]
        text = load_pdf(pdf_path)
        chunks = split_text(text)

        print("PDF:", pdf_path.name)
        print("Extracted characters:", len(text))
        print("Total chunks:", len(chunks))

        if chunks:
            print("\nFirst chunk:\n")
            print(chunks[0])
