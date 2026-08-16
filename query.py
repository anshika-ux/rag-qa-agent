import sys
import chromadb
from chromadb.utils import embedding_functions
import requests

CHROMA_DIR = "chroma_db"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"   # run `ollama pull llama3.2` first


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name="documents", embedding_function=embed_fn
    )


def retrieve(query, n_results=4):
    """Return top-n relevant chunks for a query, with their sources."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    return list(zip(chunks, metadatas))


def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(
        f"[Source: {meta['source']}, chunk {meta['chunk']}]\n{chunk}"
        for chunk, meta in retrieved_chunks
    )
    prompt = f"""You are a helpful assistant answering questions using ONLY the context below.
If the answer isn't in the context, say you don't know — don't make things up.

Context:
{context}

Question: {query}

Answer:"""
    return prompt


def ask_ollama(prompt, model=OLLAMA_MODEL):
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def answer_question(query, n_results=4, verbose=True):
    retrieved = retrieve(query, n_results=n_results)

    if not retrieved:
        return "No documents found. Run ingest.py first.", []

    if verbose:
        print("\n--- Retrieved chunks ---")
        for chunk, meta in retrieved:
            print(f"  [{meta['source']} #{meta['chunk']}] {chunk[:80]}...")
        print("------------------------\n")

    prompt = build_prompt(query, retrieved)
    answer = ask_ollama(prompt)
    sources = [meta["source"] for _, meta in retrieved]
    return answer, sources


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask a question about your documents: ")

    answer, sources = answer_question(question)
    print(f"Answer: {answer}\n")
    print(f"Sources: {', '.join(set(sources))}")
