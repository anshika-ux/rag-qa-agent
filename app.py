import os
import html as html_lib

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions

from ingest import load_pdfs, chunk_text, DATA_DIR, CHROMA_DIR
from query import answer_question
from agent import run_agent
from llm import using_groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAGmate",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# ICON HELPER
# ============================================================

ICON = lambda name, size=20: (
    f'<span class="material-symbols-outlined" '
    f'style="font-size:{size}px;">{name}</span>'
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
"""<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');


/* ============================================================
   THEME
============================================================ */

:root {

    --bg: #07111F;

    --sidebar: #091523;

    --surface: #0E1B2A;

    --surface-2: #122337;

    --surface-3: #172B41;

    --text: #F5F9FC;

    --muted: #8EA3B8;

    --muted-2: #6F8499;

    --border: #20364D;

    --border-light: #29445E;

    --teal: #14B8A6;

    --teal-dark: #0F766E;

    --cyan: #22D3EE;

    --teal-soft: rgba(20,184,166,.12);

    --cyan-soft: rgba(34,211,238,.10);

    --shadow:
        0 15px 45px rgba(0,0,0,.28);
}


/* ============================================================
   GLOBAL
============================================================ */

html,
body,
.stApp {

    background: var(--bg) !important;

    color: var(--text) !important;

    font-family:
        'Inter',
        -apple-system,
        BlinkMacSystemFont,
        sans-serif !important;
}


* {
    font-family:
        'Inter',
        -apple-system,
        BlinkMacSystemFont,
        sans-serif;
}


.material-symbols-outlined {

    font-family:
        'Material Symbols Outlined' !important;

    font-weight: normal;

    font-style: normal;

    line-height: 1;

    letter-spacing: normal;

    text-transform: none;

    white-space: nowrap;

    word-wrap: normal;

    direction: ltr;

    -webkit-font-smoothing: antialiased;

    font-variation-settings:
        'FILL' 0,
        'wght' 400,
        'GRAD' 0,
        'opsz' 24;
}


/* ============================================================
   HIDE STREAMLIT DEFAULT UI
============================================================ */

#MainMenu,
footer,
.stDeployButton,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {

    display: none !important;
}


header[data-testid="stHeader"] {

    background: transparent !important;
}


/* ============================================================
   MAIN AREA
============================================================ */

.block-container {

    max-width: 960px !important;

    padding-top: 1rem !important;

    padding-bottom: 10rem !important;
}


/* ============================================================
   SIDEBAR
============================================================ */

section[data-testid="stSidebar"] {

    background:
        var(--sidebar) !important;

    border-right:
        1px solid var(--border) !important;
}


section[data-testid="stSidebar"] > div {

    background:
        var(--sidebar) !important;
}


/* Sidebar buttons */

section[data-testid="stSidebar"]
.stButton
button {

    width: 100% !important;

    min-height: 42px !important;

    border-radius: 12px !important;

    background:
        transparent !important;

    border:
        1px solid var(--border) !important;

    color:
        var(--text) !important;

    font-weight: 600 !important;

    transition:
        all .18s ease !important;
}


section[data-testid="stSidebar"]
.stButton
button:hover {

    background:
        var(--surface) !important;

    border-color:
        var(--teal) !important;

    transform:
        translateY(-1px) !important;
}


/* Sidebar headings */

.sidebar-section {

    margin:
        24px 0 10px;

    color:
        #7890A7;

    font-size:
        .7rem;

    font-weight:
        700;

    text-transform:
        uppercase;

    letter-spacing:
        .09em;
}


/* Sidebar file */

.file-pill {

    display:
        flex;

    align-items:
        center;

    gap:
        10px;

    width:
        100%;

    box-sizing:
        border-box;

    padding:
        10px 11px;

    margin-bottom:
        7px;

    background:
        rgba(255,255,255,.018);

    border:
        1px solid var(--border);

    border-radius:
        12px;

    color:
        var(--text);

    font-size:
        .8rem;

    overflow:
        hidden;

    transition:
        all .18s ease;
}


.file-pill:hover {

    background:
        var(--surface);

    border-color:
        rgba(20,184,166,.45);

    transform:
        translateX(2px);
}


.file-pill .ext {

    width:
        30px;

    height:
        30px;

    min-width:
        30px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        8px;

    background:
        var(--teal-soft);

    color:
        var(--teal);
}


.file-pill .filename {

    overflow:
        hidden;

    text-overflow:
        ellipsis;

    white-space:
        nowrap;
}


/* ============================================================
   TOP BAR
============================================================ */

.topbar {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    padding:
        7px 2px 15px;

    margin-bottom:
        16px;

    border-bottom:
        1px solid rgba(32,54,77,.65);
}


.brand {

    display:
        flex;

    align-items:
        center;

    gap:
        11px;
}


.logo {

    width:
        40px;

    height:
        40px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        12px;

    background:
        linear-gradient(
            135deg,
            #14B8A6,
            #0891B2
        );

    color:
        white;

    box-shadow:
        0 8px 26px rgba(20,184,166,.22);
}


.logo .material-symbols-outlined {

    font-size:
        21px !important;

    color:
        white;
}


.brand-name {

    font-size:
        1.1rem;

    font-weight:
        700;

    letter-spacing:
        -.025em;
}


.topbar-right {

    display:
        flex;

    align-items:
        center;

    gap:
        8px;
}


.doc-chip,
.model-chip {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        6px;

    padding:
        6px 11px;

    border-radius:
        999px;

    background:
        var(--surface);

    border:
        1px solid var(--border);

    color:
        var(--muted);

    font-size:
        .72rem;

    font-weight:
        600;
}


.doc-chip:not(.empty) {

    background:
        var(--teal-soft);

    border-color:
        rgba(20,184,166,.25);

    color:
        #7DE8D5;
}


.model-chip .dot {

    width:
        7px;

    height:
        7px;

    border-radius:
        50%;

    background:
        var(--teal);

    box-shadow:
        0 0 0 4px var(--teal-soft);
}


/* ============================================================
   HERO
============================================================ */

.hero {

    text-align:
        center;

    padding:
        2rem .5rem 1rem;
}


.hero-icon {

    width:
        52px;

    height:
        52px;

    margin:
        0 auto 17px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        16px;

    background:
        linear-gradient(
            135deg,
            rgba(20,184,166,.14),
            rgba(34,211,238,.08)
        );

    border:
        1px solid rgba(20,184,166,.18);

    color:
        var(--teal);
}


.hero h1 {

    margin:
        0 0 11px;

    font-size:
        2.45rem;

    font-weight:
        700;

    line-height:
        1.1;

    letter-spacing:
        -.05em;

    background:
        linear-gradient(
            90deg,
            #F8FAFC 15%,
            #A5F3FC 90%
        );

    -webkit-background-clip:
        text;

    background-clip:
        text;

    -webkit-text-fill-color:
        transparent;
}


.hero p {

    max-width:
        540px;

    margin:
        0 auto;

    color:
        var(--muted);

    font-size:
        .91rem;

    line-height:
        1.7;
}


/* ============================================================
   SUGGESTION BUTTONS
============================================================ */

.suggest-grid {

    margin:
        22px auto 0;
}


.suggest-grid .stButton {

    margin-bottom:
        10px;
}


.suggest-grid .stButton button {

    position:
        relative !important;

    width:
        100% !important;

    min-height:
        76px !important;

    padding:
        13px 17px !important;

    border:
        1px solid var(--border) !important;

    border-radius:
        15px !important;

    background:
        var(--surface) !important;

    color:
        var(--text) !important;

    font-size:
        .87rem !important;

    font-weight:
        600 !important;

    white-space:
        pre-line !important;

    transition:
        all .2s ease !important;
}


.suggest-grid .stButton button:hover {

    background:
        var(--surface-2) !important;

    border-color:
        rgba(20,184,166,.58) !important;

    transform:
        translateY(-3px) !important;

    box-shadow:
        0 13px 30px rgba(0,0,0,.24) !important;
}


/* ============================================================
   CHAT
============================================================ */

div[data-testid="stChatMessage"] {

    background:
        transparent !important;

    border:
        none !important;

    padding:
        .35rem 0 !important;

    animation:
        messageIn .25s ease-out;
}


@keyframes messageIn {

    from {

        opacity:
            0;

        transform:
            translateY(8px);
    }

    to {

        opacity:
            1;

        transform:
            translateY(0);
    }
}


[data-testid="stChatMessageAvatarUser"] {

    background:
        #334155 !important;
}


[data-testid="stChatMessageAvatarAssistant"] {

    background:
        linear-gradient(
            135deg,
            var(--teal),
            #0891B2
        ) !important;
}


/* User bubble */

.user-bubble {

    display:
        inline-block;

    max-width:
        100%;

    box-sizing:
        border-box;

    padding:
        12px 17px;

    background:
        linear-gradient(
            135deg,
            #13243A,
            #102033
        );

    border:
        1px solid var(--border-light);

    border-radius:
        18px;

    line-height:
        1.6;

    color:
        var(--text);
}


/* Sources */

.source-row {

    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        5px;

    margin-top:
        12px;
}


.source-tag {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        5px;

    padding:
        5px 10px;

    border-radius:
        999px;

    background:
        var(--cyan-soft);

    border:
        1px solid rgba(34,211,238,.2);

    color:
        #8DECF3;

    font-size:
        .69rem;
}


/* ============================================================
   THINKING
============================================================ */

.thinking {

    display:
        flex;

    align-items:
        center;

    gap:
        10px;

    color:
        var(--muted);

    font-size:
        .87rem;
}


.thinking-dots {

    display:
        inline-flex;

    align-items:
        center;
}


.thinking-dots span {

    display:
        inline-block;

    width:
        6px;

    height:
        6px;

    margin-right:
        3px;

    border-radius:
        50%;

    background:
        var(--teal);

    animation:
        pulse 1.1s infinite ease-in-out;
}


.thinking-dots span:nth-child(2) {

    animation-delay:
        .14s;
}


.thinking-dots span:nth-child(3) {

    animation-delay:
        .28s;
}


@keyframes pulse {

    0%,
    80%,
    100% {

        opacity:
            .35;

        transform:
            scale(.65);
    }

    40% {

        opacity:
            1;

        transform:
            scale(1);
    }
}


/* ============================================================
   TOOL TRACE
============================================================ */

.tool-trace {

    padding:
        11px 13px;

    margin-bottom:
        8px;

    background:
        #0A1726;

    border:
        1px solid var(--border);

    border-radius:
        11px;

    color:
        var(--muted);

    font-size:
        .78rem;

    line-height:
        1.5;

    white-space:
        pre-wrap;
}


.tool-trace .tool-name {

    color:
        var(--cyan);

    font-weight:
        600;
}


/* ============================================================
   EXPANDER
============================================================ */

div[data-testid="stExpander"] {

    background:
        #0B1928 !important;

    border:
        1px solid var(--border) !important;

    border-radius:
        13px !important;
}


/* ============================================================
   GENERAL BUTTONS
============================================================ */

.stButton button {

    border-radius:
        12px !important;

    border:
        1px solid var(--border) !important;

    background:
        var(--surface) !important;

    color:
        var(--text) !important;

    transition:
        all .18s ease !important;
}


.stButton button:hover {

    background:
        var(--surface-2) !important;

    border-color:
        var(--teal) !important;
}


/* ============================================================
   COMPOSER CONTAINER
============================================================ */

.st-key-composer {

    position:
        fixed !important;

    left:
        0 !important;

    right:
        0 !important;

    bottom:
        0 !important;

    z-index:
        999 !important;

    width:
        100% !important;

    box-sizing:
        border-box !important;

    padding:
        0 1rem 1rem !important;

    pointer-events:
        none !important;

    background:
        linear-gradient(
            to top,
            var(--bg) 65%,
            transparent
        ) !important;
}


.st-key-composer > div {

    max-width:
        900px !important;

    margin:
        0 auto !important;

    pointer-events:
        auto !important;
}


/* File chips */

.st-key-composer .msg-bar-files {

    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        6px;

    margin-bottom:
        8px;
}


.msg-file-chip {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        6px;

    max-width:
        100%;

    padding:
        5px 10px;

    background:
        rgba(14,27,42,.96);

    border:
        1px solid var(--border);

    border-radius:
        999px;

    color:
        var(--muted);

    font-size:
        .71rem;
}


/* ============================================================
   COMPOSER INPUT
============================================================ */

.st-key-composer
[data-testid="stTextInput"] {

    margin:
        0 !important;
}


.st-key-composer
[data-testid="stTextInput"]
> div
> div {

    background:
        transparent !important;

    border:
        none !important;

    box-shadow:
        none !important;
}


.st-key-composer
[data-testid="stTextInput"]
input {

    height:
        42px !important;

    background:
        transparent !important;

    border:
        none !important;

    color:
        var(--text) !important;

    font-size:
        .93rem !important;

    padding:
        8px 4px !important;
}


.st-key-composer
[data-testid="stTextInput"]
input::placeholder {

    color:
        var(--muted-2) !important;
}


.st-key-composer
[data-testid="stTextInput"]
input:focus {

    outline:
        none !important;

    box-shadow:
        none !important;
}


.st-key-composer
[data-testid="stTextInput"]
label {

    display:
        none !important;
}


/* ============================================================
   ATTACH BUTTON
============================================================ */

.st-key-composer
[data-testid="stPopover"]
> button {

    width:
        42px !important;

    min-width:
        42px !important;

    height:
        42px !important;

    min-height:
        42px !important;

    padding:
        0 !important;

    flex-shrink:
        0 !important;

    border:
        1px solid transparent !important;

    border-radius:
        50% !important;

    background:
        transparent !important;

    color:
        var(--muted) !important;

    font-size:
        0 !important;

    white-space:
        nowrap !important;
}


.st-key-composer
[data-testid="stPopover"]
> button::before {

    content:
        "attach_file";

    font-family:
        'Material Symbols Outlined';

    font-size:
        21px;

    font-variation-settings:
        'FILL' 0,
        'wght' 400,
        'GRAD' 0,
        'opsz' 24;
}


.st-key-composer
[data-testid="stPopover"]
> button:hover {

    background:
        var(--surface-3) !important;

    color:
        var(--text) !important;
}


/* ============================================================
   CHAT INPUT (replaces old form + text_input + submit button)
============================================================ */

.st-key-composer
[data-testid="stChatInput"] {

    background:
        transparent !important;

    border:
        none !important;
}


.st-key-composer
[data-testid="stChatInput"]
textarea {

    background:
        transparent !important;

    border:
        none !important;

    color:
        var(--text) !important;

    font-size:
        .93rem !important;
}


.st-key-composer
[data-testid="stChatInput"]
textarea::placeholder {

    color:
        var(--muted-2) !important;
}


.st-key-composer
[data-testid="stChatInput"]
button {

    background:
        linear-gradient(
            135deg,
            var(--teal),
            #0891B2
        ) !important;

    border-radius:
        50% !important;

    box-shadow:
        0 5px 18px rgba(20,184,166,.18);

    transition:
        all .18s ease !important;
}


.st-key-composer
[data-testid="stChatInput"]
button:hover {

    transform:
        scale(1.06);

    box-shadow:
        0 7px 25px rgba(20,184,166,.32);
}


.st-key-composer
[data-testid="stChatInput"]
button:disabled {

    background:
        #1D3042 !important;

    box-shadow:
        none !important;
}


/* ============================================================
   COMPOSER INNER CARD
============================================================ */

.st-key-composer
[data-testid="stHorizontalBlock"] {

    align-items:
        center !important;
}


.st-key-composer
[data-testid="stHorizontalBlock"]:last-child {

    padding:
        5px 7px !important;

    background:
        rgba(14,27,42,.98) !important;

    border:
        1px solid var(--border-light) !important;

    border-radius:
        23px !important;

    box-shadow:
        var(--shadow) !important;
}


/* ============================================================
   UPLOAD POPOVER
============================================================ */

[data-testid="stPopoverBody"] {

    background:
        #0D1B2A !important;

    border:
        1px solid var(--border) !important;

    border-radius:
        14px !important;
}


[data-testid="stPopoverBody"]
[data-testid="stFileUploader"] {

    margin-top:
        8px;
}


/* ============================================================
   MOBILE
============================================================ */

@media (max-width: 700px) {

    .block-container {

        padding-left:
            .75rem !important;

        padding-right:
            .75rem !important;
    }


    .hero {

        padding-top:
            1.5rem;
    }


    .hero h1 {

        font-size:
            1.75rem;
    }


    .topbar-right {

        display:
            none;
    }


    .suggest-grid
    .stButton
    button {

        min-height:
            68px !important;
    }
}

</style>""",
    unsafe_allow_html=True,
)


# ============================================================
# SUGGESTIONS
# ============================================================

SUGGESTIONS = [
    (
        "📄  Summarize\n      Get a quick overview",
        "Summarize",
        "Summarize the key points in my documents.",
    ),
    (
        "💡  Explain\n      Break down complex topics",
        "Explain",
        "Explain the most important ideas in simple language.",
    ),
    (
        "🔎  Find gaps\n      Spot what's missing",
        "Find gaps",
        "What questions do my documents leave unanswered?",
    ),
    (
        "🌐  Web check\n      Cross-reference online",
        "Web check",
        "Search the web for recent context related to my documents.",
    ),
]


# ============================================================
# FILE HELPERS
# ============================================================

def list_pdfs():
    return sorted(
        f
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    )


# ============================================================
# INGEST DOCUMENTS
# ============================================================

def ingest_documents():

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    embed_fn = (
        embedding_functions
        .SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    try:
        client.delete_collection(
            "documents"
        )
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=embed_fn,
    )

    docs = load_pdfs(DATA_DIR)

    if not docs:
        return 0, 0

    total_chunks = 0

    for fname, text in docs:

        chunks = chunk_text(text)

        ids = [
            f"{fname}-{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": fname,
                "chunk": i,
            }
            for i in range(len(chunks))
        ]

        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)

    return len(docs), total_chunks


# ============================================================
# UPLOAD HANDLER
# ============================================================

def handle_upload(
    uploaded_files,
    clear_old=True,
):

    if not uploaded_files:
        return None

    upload_key = tuple(
        sorted(
            f.name
            for f in uploaded_files
        )
    )

    if (
        st.session_state.last_upload
        == upload_key
    ):
        return "ready"

    if clear_old:

        for old_f in list_pdfs():

            os.remove(
                os.path.join(
                    DATA_DIR,
                    old_f,
                )
            )

    for f in uploaded_files:

        with open(
            os.path.join(
                DATA_DIR,
                f.name,
            ),
            "wb",
        ) as out:

            out.write(
                f.getbuffer()
            )

    with st.spinner(
        "Reading your documents..."
    ):

        n_docs, _ = ingest_documents()

    st.session_state.last_upload = upload_key

    if n_docs:
        return f"indexed:{n_docs}"

    return "empty"


# ============================================================
# ASSISTANT MESSAGE
# ============================================================

def render_assistant_message(item):

    st.markdown(
        item["content"]
    )

    sources = item.get(
        "sources"
    ) or []

    if sources:

        tags = "".join(
            f"""
            <span class="source-tag">
                📄 {html_lib.escape(s)}
            </span>
            """
            for s in sources
        )

        st.html(
            f"""
            <div class="source-row">
                {tags}
            </div>
            """
        )

    tool_steps = item.get(
        "tool_steps"
    ) or []

    if tool_steps:

        with st.expander(
            "How this was answered",
            expanded=False,
        ):

            for t in tool_steps:

                preview = (
                    t["result"][:400]
                    + (
                        "..."
                        if len(t["result"]) > 400
                        else ""
                    )
                )

                st.html(
                    f"""
                    <div class="tool-trace">

                        <span class="tool-name">
                            {html_lib.escape(
                                str(t["tool"])
                            )}
                        </span>

                        <br>

                        {html_lib.escape(
                            str(t["args"])
                        )}

                        <div>
                            {html_lib.escape(
                                preview
                            )}
                        </div>

                    </div>
                    """
                )


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "agent_toggle" not in st.session_state:
    st.session_state.agent_toggle = True

if "last_upload" not in st.session_state:
    st.session_state.last_upload = None

if "clear_old" not in st.session_state:
    st.session_state.clear_old = True


# ============================================================
# DOCUMENTS
# ============================================================

existing = list_pdfs()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if st.button(
        "＋  New chat",
        use_container_width=True,
    ):

        st.session_state.history = []

        st.session_state.pending_question = None

        st.rerun()


    st.html(
        f"""
        <div class="sidebar-section">
            {ICON("folder", 14)}
            Your library
        </div>
        """
    )


    existing = list_pdfs()


    if existing:

        for f in existing:

            st.html(
                f"""
                <div class="file-pill">

                    <div class="ext">
                        {ICON("picture_as_pdf", 15)}
                    </div>

                    <div class="filename">
                        {html_lib.escape(f)}
                    </div>

                </div>
                """
            )

    else:

        st.caption(
            "No documents yet."
        )

        st.caption(
            "Use the 📎 button below to upload a PDF."
        )


    st.divider()


    with st.expander(
        "⚙  Settings"
    ):

        st.toggle(
            "Smart agent mode",
            key="agent_toggle",
            help=(
                "Search documents, web, "
                "or calculate automatically."
            ),
        )


        backend = (
            "Cloud"
            if using_groq()
            else "Local"
        )

        st.caption(
            f"Backend  ·  {backend}"
        )


        if st.button(
            "↻  Re-index documents",
            use_container_width=True,
        ):

            with st.spinner(
                "Re-indexing..."
            ):

                n_docs, _ = ingest_documents()


            if n_docs:

                st.success(
                    f"{n_docs} document(s) indexed."
                )

            else:

                st.warning(
                    "Nothing to index."
                )


# ============================================================
# TOP BAR
# ============================================================

doc_label = (

    f"{len(existing)} PDF"
    f"{'s' if len(existing) != 1 else ''}"

    if existing

    else

    "No documents"
)


doc_class = (
    ""
    if existing
    else
    "empty"
)


st.html(
    f"""
    <div class="topbar">

        <div class="brand">

            <div class="logo">
                {ICON("auto_stories", 21)}
            </div>

            <div class="brand-name">
                RAGmate
            </div>

        </div>


        <div class="topbar-right">

            <div class="doc-chip {doc_class}">
                {ICON("description", 14)}
                {doc_label}
            </div>

            <div class="model-chip">
                <span class="dot"></span>
                {ICON("smart_toy", 14)}
                Agent
            </div>

        </div>

    </div>
    """
)


# ============================================================
# CHAT HISTORY
# ============================================================

for item in st.session_state.history:

    with st.chat_message(
        item["role"]
    ):

        if item["role"] == "user":

            st.html(
                f"""
                <div class="user-bubble">
                    {html_lib.escape(
                        item["content"]
                    )}
                </div>
                """
            )

        else:

            render_assistant_message(
                item
            )


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.history:

    st.html(
        f"""
        <div class="hero">

            <div class="hero-icon">
                {ICON("auto_awesome", 25)}
            </div>

            <h1>
                Ask your documents anything.
            </h1>

            <p>
                Upload a PDF, retrieve the most relevant
                context, and get grounded answers powered
                by your AI.
            </p>

        </div>
        """
    )


    st.markdown(
        '<div class="suggest-grid">',
        unsafe_allow_html=True,
    )


    row1 = st.columns(
        2,
        gap="medium",
    )

    row2 = st.columns(
        2,
        gap="medium",
    )


    suggestion_columns = [
        row1[0],
        row1[1],
        row2[0],
        row2[1],
    ]


    for col, (
        label,
        key_name,
        prompt,
    ) in zip(
        suggestion_columns,
        SUGGESTIONS,
    ):

        with col:

            if st.button(
                label,
                key=f"suggest_{key_name}",
                use_container_width=True,
            ):

                st.session_state.pending_question = prompt

                st.rerun()


    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# COMPOSER
# ============================================================

with st.container(
    key="composer"
):

    existing = list_pdfs()


    # --------------------------------------------------------
    # FILE CHIPS
    # --------------------------------------------------------

    if existing:

        file_chips = "".join(

            f"""
            <span class="msg-file-chip">

                {ICON("picture_as_pdf", 13)}

                {html_lib.escape(f)}

            </span>
            """

            for f in existing
        )


        st.html(
            f"""
            <div class="msg-bar-files">
                {file_chips}
            </div>
            """
        )


    # --------------------------------------------------------
    # ATTACH (outside any form so uploads process immediately,
    # instead of waiting for the next form submission)
    # --------------------------------------------------------

    attach_col, input_col = st.columns(
        [0.07, 0.93],
        gap="small",
        vertical_alignment="center",
    )

    with attach_col:

        with st.popover(
            "📎",
            help="Attach PDF",
        ):

            st.markdown(
                "### Add documents"
            )

            st.caption(
                "Upload one or more PDF files."
            )

            st.checkbox(
                "Replace previous files",
                key="clear_old",
            )

            bar_upload = st.file_uploader(
                "Choose PDF",
                type=["pdf"],
                accept_multiple_files=True,
                key="bar_uploader",
                label_visibility="collapsed",
            )

            upload_status = handle_upload(
                bar_upload,
                clear_old=st.session_state.get(
                    "clear_old",
                    True,
                ),
            )

            if (
                upload_status
                and upload_status.startswith(
                    "indexed:"
                )
            ):

                n = upload_status.split(
                    ":"
                )[1]

                st.success(
                    f"{n} file"
                    f"{'s' if n != '1' else ''}"
                    " ready"
                )

                st.rerun()

            elif upload_status == "empty":

                st.warning(
                    "Couldn't read that PDF."
                )

    # --------------------------------------------------------
    # INPUT (native chat_input: handles Enter-to-send on its own,
    # no form needed, so uploads above are never trapped waiting
    # for a submit event)
    # --------------------------------------------------------

    with input_col:

        chat_submission = st.chat_input(
            "Ask anything about your documents...",
            key="message_input",
        )


# ============================================================
# HANDLE QUESTION
# ============================================================

question = st.session_state.pending_question


if chat_submission and chat_submission.strip():

    question = chat_submission.strip()

    st.session_state.pending_question = None

elif question:

    st.session_state.pending_question = None


existing = list_pdfs()


# ============================================================
# RUN RAG / AGENT
# ============================================================

if question:

    needs_pdf = not st.session_state.get("agent_toggle", True)

    if needs_pdf and not existing:

        st.toast(
            "Attach a PDF first using the 📎 button."
        )

    else:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        st.session_state.history.append(
            {
                "role": "user",
                "content": question,
            }
        )


        with st.chat_message(
            "user"
        ):

            st.html(
                f"""
                <div class="user-bubble">
                    {html_lib.escape(question)}
                </div>
                """
            )


        # ----------------------------------------------------
        # ASSISTANT
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            thinking = st.empty()


            thinking.html(
                """
                <div class="thinking">

                    <span class="thinking-dots">

                        <span></span>
                        <span></span>
                        <span></span>

                    </span>

                    Thinking...

                </div>
                """
            )


            tool_steps = []

            sources = []


            try:

                if st.session_state.get(
                    "agent_toggle",
                    True,
                ):

                    answer, trace = run_agent(
                        question,
                        verbose=False,
                    )


                    tool_steps = [
                        t
                        for t in trace
                        if t["type"]
                        == "tool_call"
                    ]


                else:

                    answer, raw = answer_question(
                        question,
                        verbose=False,
                    )


                    sources = sorted(
                        set(raw)
                    )


            except Exception as e:

                answer = (
                    "Something went wrong. "
                    "Make sure your model is running "
                    "and a PDF is attached."
                )


                if os.environ.get(
                    "MARGINALIA_DEBUG"
                ):

                    answer += (
                        f"\n\n({e})"
                    )


            thinking.empty()


            assistant_item = {
                "role": "assistant",
                "content": answer,
                "tool_steps": tool_steps,
                "sources": sources,
            }


            render_assistant_message(
                assistant_item
            )


        st.session_state.history.append(
            assistant_item
        )


        st.rerun()