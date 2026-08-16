import os
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"


def load_pdfs(data_dir):
    """Read all PDFs in data/ and return list of (filename, full_text)"""
    docs = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".pdf"):
            path = os.path.join(data_dir, fname)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            docs.append((fname, text))
    return docs


def chunk_text(text, chunk_size=800, overlap=150):
    """Simple sliding-window chunking by characters"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="documents", embedding_function=embed_fn
    )

    docs = load_pdfs(DATA_DIR)
    print(f"Loaded {len(docs)} PDF(s)")

    if not docs:
        print(f"No PDFs found in {DATA_DIR}/. Add some PDFs and re-run.")
        return

    for fname, text in docs:
        chunks = chunk_text(text)
        ids = [f"{fname}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": fname, "chunk": i} for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        print(f"  {fname}: {len(chunks)} chunks added")

    print("Ingestion complete. Vector DB stored in ./chroma_db")


if __name__ == "__main__":
    main()
