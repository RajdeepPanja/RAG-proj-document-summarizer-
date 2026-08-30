import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------
# Config
# ----------------------------
PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="The Archive · Chat with your book", page_icon="📖", layout="wide")

# ----------------------------
# Styling — "The Archive": a night-library aesthetic
# deep forest ink, brass/gilt accent, parchment for message bubbles
# ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;1,500&family=Source+Sans+3:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ink: #0f1b14;
        --panel: #16261c;
        --panel-edge: #2a3d2c;
        --parchment: #ede6d6;
        --gold: #c9a227;
        --gold-soft: #8a7328;
        --text-light: #ece7d8;
        --text-muted: #9caf9f;
    }

    .stApp {
        background: radial-gradient(ellipse at top left, #142218 0%, #0f1b14 55%, #0b140e 100%);
        font-family: 'Source Sans 3', sans-serif;
    }

    /* Force light, readable text everywhere in the main content area.
       Streamlit's own components ship inline/high-specificity dark text
       colors, which is why text disappears once the background goes dark. */
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
    .stApp div, .stMarkdown, [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
    [data-testid="stChatMessageContent"], [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-light) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10190f 0%, #0d150c 100%);
        border-right: 1px solid var(--panel-edge);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-light) !important;
    }
    section[data-testid="stSidebar"] h2 {
        font-family: 'Playfair Display', serif;
        letter-spacing: 0.04em;
        color: var(--gold) !important;
        border-bottom: 1px solid var(--gold-soft);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    /* "Browse files" button inside the uploader keeps its own dark-on-light
       styling in some Streamlit themes — force it to match the dark panel. */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button * {
        background: var(--panel) !important;
        color: var(--text-light) !important;
        border: 1px solid var(--gold-soft) !important;
    }
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span {
        color: var(--text-muted) !important;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Header block */
    .archive-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 1.2rem 0 1.6rem 0;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid var(--panel-edge);
    }
    .archive-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.32em;
        text-transform: uppercase;
        color: var(--gold-soft);
        margin-bottom: 0.35rem;
    }
    .archive-title {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 2.6rem;
        color: var(--parchment);
        margin: 0;
        letter-spacing: 0.01em;
    }
    .archive-title em {
        color: var(--gold);
        font-style: italic;
    }
    .archive-rule {
        width: 80px;
        height: 2px;
        margin-top: 0.9rem;
        background: linear-gradient(90deg, transparent, var(--gold), transparent);
    }
    .archive-sub {
        font-family: 'Source Sans 3', sans-serif;
        color: var(--text-muted);
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }

    /* Volume badge in sidebar */
    .volume-card {
        border: 1px dashed var(--gold-soft);
        border-radius: 6px;
        padding: 0.7rem 0.9rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        background: rgba(201, 162, 39, 0.06);
        margin-top: 0.4rem;
    }
    .volume-label {
        color: var(--gold) !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-size: 0.68rem;
        display: block;
        margin-bottom: 0.3rem;
    }

    /* Buttons — dark ink text on the gold gradient for contrast */
    .stButton > button {
        background: linear-gradient(180deg, #d4b23c, #b3901f);
        color: #16210f !important;
        border: none;
        border-radius: 5px;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 0.5rem 1rem;
        transition: filter 0.15s ease, transform 0.15s ease;
    }
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #16210f !important;
    }
    .stButton > button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(201, 162, 39, 0.05) !important;
        border: 1px dashed var(--gold-soft) !important;
        border-radius: 8px !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 0.4rem 0.2rem;
        margin-bottom: 0.4rem;
    }
    div[data-testid="stChatMessageContent"] {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.98rem;
        line-height: 1.55;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: var(--panel) !important;
        color: var(--text-light) !important;
        border: 1px solid var(--panel-edge) !important;
    }
    [data-testid="stChatInput"] {
        border-top: 1px solid var(--panel-edge);
    }

    /* Alerts — explicit dark panel + colored left border per type, so
       contrast never depends on Streamlit's own light-theme alert colors */
    div[data-testid="stAlert"] {
        border-radius: 8px;
        font-family: 'Source Sans 3', sans-serif;
        background: var(--panel) !important;
        border: 1px solid var(--panel-edge);
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] div {
        color: var(--text-light) !important;
    }
    div[data-testid="stAlertContainer"] {
        background: var(--panel) !important;
    }

    /* Divider */
    hr {
        border-color: var(--panel-edge) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Prompt template (same as main.py)
# ----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

        Use ONLY the provided context to answer the question.

        If the answer is not present in the context,
        say: "I could not find the answer in the document."
        """,
        ),
        (
            "human",
            """Context:
            {context}

            Question:
            {question}
            """,
        ),
    ]
)


# ----------------------------
# Cached resources
# ----------------------------
@st.cache_resource
def get_embedding_model():
    return MistralAIEmbeddings(model="mistral-embed")


@st.cache_resource
def get_llm():
    return ChatMistralAI(model="mistral-small-2603")


def build_vectorstore_from_pdf(pdf_path: str, persist_directory: str):
    """Load a PDF, chunk it, embed it, and persist it to Chroma."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.page_content = chunk.page_content.encode(
            "ascii", errors="ignore"
        ).decode("ascii")

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    return vectorstore, len(chunks)


def load_existing_vectorstore(persist_directory: str):
    embedding_model = get_embedding_model()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )


def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )


def answer_question(retriever, llm, question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(final_prompt)
    return response.content


# ----------------------------
# Session state init
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None

# If a database already exists on disk from a previous run, offer to load it
if st.session_state.vectorstore is None and os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    try:
        st.session_state.vectorstore = load_existing_vectorstore(PERSIST_DIR)
        st.session_state.book_name = "Previously archived volume"
    except Exception:
        pass

# ----------------------------
# Sidebar: upload & process book
# ----------------------------
with st.sidebar:
    st.markdown("## 📖 The Archive")
    st.caption("Bring a book. Ask it anything.")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("📜  Process & Create Database", type="primary", use_container_width=True):
            with st.spinner("Reading pages, weaving embeddings..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    vectorstore, num_chunks = build_vectorstore_from_pdf(
                        tmp_path, PERSIST_DIR
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.book_name = uploaded_file.name
                    st.session_state.messages = []  # reset chat for new book
                    st.success(f"Catalogued {num_chunks} passages.")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                finally:
                    os.remove(tmp_path)

    st.markdown("<hr/>", unsafe_allow_html=True)

    if st.session_state.book_name:
        st.markdown(
            f"""<div class="volume-card">
                <span class="volume-label">Active Volume</span>
                {st.session_state.book_name}
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info("No volume archived yet.")

    if st.session_state.vectorstore is not None:
        st.write("")
        if st.button("🗞️  Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ----------------------------
# Main: header + chat interface
# ----------------------------
st.markdown(
    """
    <div class="archive-header">
        <div class="archive-eyebrow">Retrieval-Augmented Reading Room</div>
        <div class="archive-title">The <em>Archive</em></div>
        <div class="archive-rule"></div>
        <div class="archive-sub">Upload a book on the left, then converse with it here.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF book from the sidebar and click **Process & Create Database** to get started.")
else:
    # Show chat history
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "📖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Chat input
    user_query = st.chat_input("Ask a question about your book...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="📖"):
            with st.spinner("Turning pages..."):
                llm = get_llm()
                retriever = get_retriever(st.session_state.vectorstore)
                answer = answer_question(retriever, llm, user_query)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
