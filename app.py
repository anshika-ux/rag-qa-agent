import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

from ingest import load_pdfs, chunk_text, DATA_DIR, CHROMA_DIR
from query import answer_question
from agent import run_agent
from llm import using_groq

st.set_page_config(page_title="Marginalia — Document Q&A Agent", page_icon="📑", layout="wide")

os.makedirs(DATA_DIR, exist_ok=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink: #0F1419;
    --surface: #161B22;
    --surface-2: #1D2530;
    --amber: #E8A33D;
    --teal: #5FB3B3;
    --text: #E6E8EB;
    --muted: #8B95A1;
    --border: #2A313C;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero header */
.app-hero {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 2px;
}
.app-hero .mark {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.4rem;
    color: var(--text);
    letter-spacing: -0.01em;
}
.app-hero .mark .hl {
    background: linear-gradient(180deg, transparent 60%, rgba(232,163,61,0.35) 60%);
    padding: 0 2px;
}
.app-tagline {
    font-family: 'Inter', sans-serif;
    color: var(--muted);
    font-size: 0.95rem;
    margin-bottom: 1.6rem;
    margin-top: 4px;
}

/* Stat cards */
.stat-row { display: flex; gap: 12px; margin-bottom: 1.8rem; }
.stat-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
}
.stat-card .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
}
.stat-card .value {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    color: var(--text);
    margin-top: 2px;
}
.stat-card.active .value { color: var(--amber); }
.stat-card.agent .value { color: var(--teal); }

/* Section eyebrow labels */
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--amber);
    margin-bottom: 2px;
}

/* Sidebar tightening */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }

/* Tool trace styling inside expander */
.tool-trace {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    background: var(--ink);
    border: 1px solid var(--border);
    border-left: 3px solid var(--teal);
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.tool-trace .tool-name { color: var(--teal); font-weight: 500; }
.tool-trace .tool-result { color: var(--muted); white-space: pre-wrap; margin-top: 4px; }

.source-tag {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: rgba(232,163,61,0.12);
    color: var(--amber);
    border: 1px solid rgba(232,163,61,0.3);
    border-radius: 4px;
    padding: 2px 8px;
    margin-top: 6px;
    margin-right: 4px;
}

/* Chat avatars */
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, var(--amber), #c9822a) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, var(--teal), #3d7f7f) !important;
}

/* Chat bubbles: subtle entrance + card feel */
[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 4px 6px;
    animation: fadeSlideIn 0.35s ease-out;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Thinking indicator */
.thinking-row {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    font-size: 0.85rem;
    padding: 6px 2px;
}
.thinking-dots span {
    display: inline-block;
    width: 6px;
    height: 6px;
    margin-right: 3px;
    border-radius: 50%;
    background: var(--teal);
    animation: bounce 1.1s infinite ease-in-out both;
}
.thinking-dots span:nth-child(1) { animation-delay: -0.28s; }
.thinking-dots span:nth-child(2) { animation-delay: -0.14s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
    40% { transform: scale(1); opacity: 1; }
}

/* Button polish */
.stButton > button {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(232,163,61,0.25);
}
.stButton > button:active { transform: translateY(0); }

/* Stat cards: gentle lift-in */
.stat-card {
    animation: fadeSlideIn 0.4s ease-out;
}

/* Hero mark: soft pulse on the icon */
.app-hero .mark { animation: fadeSlideIn 0.5s ease-out; }
</style>
""", unsafe_allow_html=True)

# ---- Hero ----
st.markdown("""
<div class="app-hero">
    <span class="mark">📑 <span class="hl">Marginalia</span></span>
</div>
<div class="app-tagline">An agent that reads your documents, searches the live web, and shows its reasoning — free and local.</div>
""", unsafe_allow_html=True)

# ---- Sidebar: upload + ingest ----
with st.sidebar:
    st.markdown('<div class="eyebrow">01 — Library</div>', unsafe_allow_html=True)
    st.markdown("### Add documents")
    clear_old = st.checkbox(
        "Replace existing documents with this upload",
        value=True,
        help="If checked, old PDFs are removed so only your latest upload(s) are searchable.",
    )
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded_files:
        if clear_old:
            for old_f in os.listdir(DATA_DIR):
                if old_f.endswith(".pdf"):
                    os.remove(os.path.join(DATA_DIR, old_f))
        for f in uploaded_files:
            save_path = os.path.join(DATA_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s)")

    if st.button("Ingest / Re-index", type="primary", use_container_width=True):
        with st.spinner("Chunking and embedding..."):
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
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
                total_chunks = 0
                for fname, text in docs:
                    chunks = chunk_text(text)
                    ids = [f"{fname}-{i}" for i in range(len(chunks))]
                    metadatas = [
                        {"source": fname, "chunk": i} for i in range(len(chunks))
                    ]
                    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
                    total_chunks += len(chunks)
                st.session_state["last_chunk_count"] = total_chunks
                st.success(f"Indexed {len(docs)} document(s), {total_chunks} chunks")

    st.markdown("---")
    st.markdown('<div class="eyebrow">02 — Mode</div>', unsafe_allow_html=True)
    agent_mode = st.toggle(
        "Agent mode",
        value=False,
        help="Off = plain RAG (always searches your documents). On = agent decides between document search, web search, and calculator.",
    )
    st.caption("🔧 Tools + web search" if agent_mode else "📄 Documents only")

    st.markdown("---")
    existing = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    st.markdown('<div class="eyebrow">03 — Library contents</div>', unsafe_allow_html=True)
    if existing:
        for f in existing:
            st.markdown(f"📄 {f}")
    else:
        st.caption("None yet — upload above.")

# ---- Stat row ----
backend_label = "Groq (cloud)" if using_groq() else "Ollama (local)"
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""<div class="stat-card"><div class="label">Documents</div><div class="value">{len(existing)}</div></div>""", unsafe_allow_html=True)
with col2:
    mode_val = "Agent" if agent_mode else "RAG"
    st.markdown(f"""<div class="stat-card {'agent' if agent_mode else ''}"><div class="label">Mode</div><div class="value">{mode_val}</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="stat-card active"><div class="label">Backend</div><div class="value" style="font-size:1.1rem;">{backend_label}</div></div>""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow" style="margin-top:1.6rem;">Ask</div>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.history:
    st.info("Upload a PDF in the sidebar, click **Ingest / Re-index**, then ask a question below.")

for role, msg in st.session_state.history:
    avatar = "🤖" if role == "assistant" else "🙂"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg)

question = st.chat_input("Ask something about your documents...")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user", avatar="🙂"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        thinking_placeholder = st.empty()
        thinking_label = "Deciding which tool to use" if agent_mode else "Reading your documents"
        thinking_placeholder.markdown(
            f"""<div class="thinking-row">
            <span class="thinking-dots"><span></span><span></span><span></span></span>
            {thinking_label}...
            </div>""",
            unsafe_allow_html=True,
        )
        try:
            if agent_mode:
                answer, trace = run_agent(question, verbose=False)
                full_answer = answer
                full_answer_html = None

                tool_steps = [t for t in trace if t["type"] == "tool_call"]
            else:
                answer, sources = answer_question(question, verbose=False)
                full_answer = answer
                tool_steps = []
                unique_sources = list(set(sources))
                if unique_sources:
                    tags = "".join(f'<span class="source-tag">{s}</span>' for s in unique_sources)
                    full_answer_html = f"{answer}<br>{tags}"
                else:
                    full_answer_html = answer
        except Exception as e:
            full_answer = (
                f"Error: {e}\n\nMake sure Ollama is running "
                "(`ollama serve` or the Ollama app) and you've ingested documents above."
            )
            full_answer_html = full_answer
            tool_steps = []

        thinking_placeholder.empty()

        if agent_mode:
            st.markdown(full_answer)
            if tool_steps:
                with st.expander(f"🔧 Agent used {len(tool_steps)} tool call(s)", expanded=False):
                    for t in tool_steps:
                        result_preview = t["result"][:400] + ("..." if len(t["result"]) > 400 else "")
                        st.markdown(
                            f"""<div class="tool-trace">
                            <span class="tool-name">{t['tool']}</span>({t['args']})
                            <div class="tool-result">{result_preview}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown(full_answer_html, unsafe_allow_html=True)

    st.session_state.history.append(("assistant", full_answer))