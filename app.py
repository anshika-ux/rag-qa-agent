import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

from ingest import load_pdfs, chunk_text, DATA_DIR, CHROMA_DIR
from query import answer_question

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

st.title("📄 RAG Document Q&A")
st.caption("Ask questions grounded in your own documents — runs 100% locally, free.")

os.makedirs(DATA_DIR, exist_ok=True)

# ---- Sidebar: upload + ingest ----
with st.sidebar:
    st.header("1. Add documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        for f in uploaded_files:
            save_path = os.path.join(DATA_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to {DATA_DIR}/")

    if st.button("Ingest / Re-index documents", type="primary"):
        with st.spinner("Chunking and embedding documents..."):
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            # reset collection so re-ingesting doesn't duplicate chunks
            try:
                client.delete_collection("documents")
            except Exception:
                pass
            collection = client.get_or_create_collection(
                name="documents", embedding_function=embed_fn
            )

            docs = load_pdfs(DATA_DIR)
            if not docs:
                st.warning("No PDFs found. Upload some first.")
            else:
                for fname, text in docs:
                    chunks = chunk_text(text)
                    ids = [f"{fname}-{i}" for i in range(len(chunks))]
                    metadatas = [
                        {"source": fname, "chunk": i} for i in range(len(chunks))
                    ]
                    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
                st.success(f"Indexed {len(docs)} document(s).")

    st.divider()
    existing = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    st.header("Documents in library")
    if existing:
        for f in existing:
            st.text(f"• {f}")
    else:
        st.text("None yet — upload above.")

# ---- Main: chat ----
st.header("2. Ask a question")

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

question = st.chat_input("Ask something about your documents...")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = answer_question(question, verbose=False)
                source_line = (
                    f"\n\n*Sources: {', '.join(set(sources))}*" if sources else ""
                )
                full_answer = answer + source_line
            except Exception as e:
                full_answer = (
                    f"Error: {e}\n\nMake sure Ollama is running "
                    "(`ollama serve` or the Ollama app) and you've run "
                    "`python ingest.py` or ingested documents above."
                )
            st.markdown(full_answer)

    st.session_state.history.append(("assistant", full_answer))
