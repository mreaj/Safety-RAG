"""
Safety Standards RAG Chatbot — Streamlit Community Cloud Edition
Stack: Mistral AI API · LangChain · ChromaDB (in-memory) · Streamlit
"""

import os, time, logging, io
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="SafetyRAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MISTRAL_MODELS = {
    "Mistral Small (fast, free tier)": "mistral-small-latest",
    "Mistral Medium":                  "mistral-medium-latest",
    "Mixtral 8x7B":                    "open-mixtral-8x7b",
    "Mistral 7B (open)":               "open-mistral-7b",
}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg:#0d0f14;--surface:#151820;--surface2:#1c2030;--border:#252a3a;
    --accent:#f0a500;--accent2:#e05c2a;--text:#e8eaf0;--muted:#6b7280;
    --success:#22c55e;--danger:#ef4444;--user-bg:#1e2540;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
.rag-header{display:flex;align-items:center;gap:14px;padding:24px 0 18px;border-bottom:1px solid var(--border);margin-bottom:20px;}
.rag-logo{width:46px;height:46px;flex-shrink:0;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 4px 18px rgba(240,165,0,.28);}
.rag-title{font-family:'DM Serif Display',serif!important;font-size:26px!important;font-weight:400!important;color:var(--text)!important;margin:0!important;}
.rag-subtitle{font-size:11px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;}
.status-badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:500;}
.status-ready{background:rgba(34,197,94,.12);color:var(--success);border:1px solid rgba(34,197,94,.25);}
.status-loading{background:rgba(240,165,0,.12);color:var(--accent);border:1px solid rgba(240,165,0,.25);}
.chat-wrapper{display:flex;flex-direction:column;gap:14px;padding:4px 0;}
.msg-user,.msg-bot{display:flex;gap:12px;align-items:flex-start;animation:fadeUp .3s ease;}
.msg-user{flex-direction:row-reverse;}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.msg-avatar{width:34px;height:34px;border-radius:9px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:15px;}
.avatar-user{background:var(--user-bg);border:1px solid var(--border);}
.avatar-bot{background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 2px 10px rgba(240,165,0,.22);}
.msg-bubble{max-width:76%;padding:13px 17px;border-radius:13px;font-size:14px;line-height:1.65;}
.bubble-user{background:var(--user-bg);border:1px solid var(--border);border-top-right-radius:4px;}
.bubble-bot{background:var(--surface);border:1px solid var(--border);border-top-left-radius:4px;}
.sources-block{margin-top:10px;padding:9px 13px;background:var(--surface2);border-left:3px solid var(--accent);border-radius:0 7px 7px 0;font-size:12px;color:var(--muted);}
.sources-block strong{color:var(--accent);}
.source-tag{display:inline-block;background:rgba(240,165,0,.08);border:1px solid rgba(240,165,0,.2);border-radius:4px;padding:2px 7px;margin:2px 2px 0 0;font-family:'JetBrains Mono',monospace;font-size:11px;color:#c9a84c;}
.stTextInput>div>div>input{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:11px!important;color:var(--text)!important;font-family:'DM Sans',sans-serif!important;font-size:14px!important;padding:13px 17px!important;transition:border-color .2s!important;}
.stTextInput>div>div>input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(240,165,0,.1)!important;}
.stTextInput>div>div>input::placeholder{color:var(--muted)!important;}
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:#0d0f14!important;border:none!important;border-radius:9px!important;font-weight:600!important;font-family:'DM Sans',sans-serif!important;transition:opacity .2s,transform .15s!important;box-shadow:0 3px 12px rgba(240,165,0,.18)!important;}
.stButton>button:hover{opacity:.87!important;transform:translateY(-1px)!important;}
.sidebar-label{font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:8px;display:block;}
.stat-row{display:flex;justify-content:space-between;margin-bottom:5px;}
.stat-key{font-size:12px;color:var(--muted);}
.stat-val{font-size:12px;font-weight:600;color:var(--text);font-family:'JetBrains Mono',monospace;}
.sidebar-section{background:var(--surface2);border:1px solid var(--border);border-radius:11px;padding:14px;margin-bottom:10px;}
[data-testid="stSelectbox"]>div>div{background:var(--surface2)!important;border-color:var(--border)!important;color:var(--text)!important;}
[data-testid="stExpander"]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:9px!important;}
[data-testid="stExpander"] summary{color:var(--muted)!important;font-size:12px!important;}
[data-testid="stFileUploader"]{background:var(--surface2)!important;border:1px dashed rgba(240,165,0,.3)!important;border-radius:11px!important;}
.welcome-card{border:1px solid var(--border);border-radius:15px;padding:28px;text-align:center;background:var(--surface);margin:32px auto;max-width:500px;}
.welcome-icon{font-size:44px;margin-bottom:14px;}
.welcome-title{font-family:'DM Serif Display',serif;font-size:21px;color:var(--text);margin-bottom:8px;}
.welcome-text{font-size:13.5px;color:var(--muted);line-height:1.6;}
.example-chip{display:inline-block;background:var(--surface2);border:1px solid var(--border);border-radius:18px;padding:5px 13px;font-size:12px;color:var(--muted);margin:3px;}
.upload-hint{background:linear-gradient(135deg,rgba(240,165,0,.06),rgba(224,92,42,.06));border:1px solid rgba(240,165,0,.2);border-radius:10px;padding:12px 16px;font-size:12px;color:#c9a84c;margin-bottom:12px;line-height:1.5;}
.api-hint{background:rgba(240,165,0,.06);border:1px solid rgba(240,165,0,.15);border-radius:9px;padding:11px 14px;font-size:12px;color:#c9a84c;line-height:1.6;margin-top:6px;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
hr{border-color:var(--border)!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stToolbar"]{display:none;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_mistral_key() -> str:
    try:
        return st.secrets["MISTRAL_API_KEY"]
    except Exception:
        return os.getenv("MISTRAL_API_KEY", "")


def format_sources(source_docs: list) -> str:
    seen, tags = set(), []
    for doc in source_docs:
        name  = doc.metadata.get("source_file", doc.metadata.get("source", "Unknown"))
        page  = doc.metadata.get("page", "")
        label = name + (f" · p{page+1}" if page != "" else "")
        if label not in seen:
            seen.add(label)
            tags.append(f'<span class="source-tag">📄 {label}</span>')
    return "".join(tags)


@st.cache_resource(show_spinner=False)
def build_vectorstore_from_bytes(_file_tuples: tuple):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
    import tempfile, docx2txt

    all_docs = []
    for filename, file_bytes in _file_tuples:
        ext = Path(filename).suffix.lower()
        try:
            if ext == ".pdf":
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(file_bytes)
                loader = PyPDFLoader(tmp.name)
                docs   = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = filename
            elif ext in (".docx", ".doc"):
                text = docx2txt.process(io.BytesIO(file_bytes))
                docs = [Document(page_content=text, metadata={"source_file": filename})]
            elif ext == ".txt":
                text = file_bytes.decode("utf-8", errors="ignore")
                docs = [Document(page_content=text, metadata={"source_file": filename})]
            else:
                continue
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"Skipped {filename}: {e}")

    if not all_docs:
        return None, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? "],
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore, len(chunks)


def build_chain(vectorstore, model_name: str, api_key: str, temperature: float, top_k: int):
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_mistralai import ChatMistralAI

    llm = ChatMistralAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=1024,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a safety standards expert assistant.
Use ONLY the context below from official safety standard documents to answer.
If the answer is not in the documents, say so clearly.
Always reference the document name when possible.

Context:
{context}

Question: {question}

Answer:""",
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 4, "lambda_mult": 0.7},
    )

    def format_docs(docs):
        return "".join(doc.page_content for doc in docs)

    # LCEL chain — modern LangChain 1.x style, no deprecated RetrievalQA
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
            source_documents=lambda x: retriever.invoke(x["question"]),
        )
        | RunnablePassthrough.assign(
            result=(prompt | llm | StrOutputParser())
        )
    )
    return chain


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("vectorstore", None),
    ("chunk_count", 0),
    ("indexed_files", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:18px 0 10px">
      <div style="font-family:'DM Serif Display',serif;font-size:20px;color:#e8eaf0">🛡️ SafetyRAG</div>
      <div style="font-size:11px;color:#6b7280;letter-spacing:.08em;text-transform:uppercase;margin-top:3px">
        Cloud Configuration
      </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # ── Mistral API Key ──
    st.markdown('<span class="sidebar-label">Mistral AI API Key</span>', unsafe_allow_html=True)
    stored_key = get_mistral_key()
    if stored_key:
        st.markdown(
            '<div style="font-size:12px;color:#22c55e;margin-bottom:8px">✅ Key loaded from secrets</div>',
            unsafe_allow_html=True,
        )
        mistral_key = stored_key
    else:
        mistral_key = st.text_input(
            "Mistral Key", type="password", placeholder="Enter your Mistral AI API key...",
            label_visibility="collapsed",
        )
        st.markdown("""
        <div class="api-hint">
          🔑 Get a free key at<br>
          <a href="https://console.mistral.ai" target="_blank" style="color:#f0a500;font-weight:600">
            console.mistral.ai
          </a><br>
          Sign up → API Keys → Create new key.<br>
          No email verification hassle!
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Model ──
    st.markdown('<span class="sidebar-label">Model</span>', unsafe_allow_html=True)
    model_label = st.selectbox("Model", list(MISTRAL_MODELS.keys()), index=0, label_visibility="collapsed")
    model_name  = MISTRAL_MODELS[model_label]
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, help="Lower = more factual")
    top_k       = st.slider("Top-K Sources", 1, 10, 5, help="Chunks retrieved per question")

    st.divider()

    # ── Upload ──
    st.markdown('<span class="sidebar-label">Upload Documents</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-hint">
      📁 Upload your safety PDFs, Word docs, or text files.<br>
      Processed in-memory — nothing stored permanently.
    </div>""", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "docs", accept_multiple_files=True,
        type=["pdf", "docx", "doc", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚙️ Build Index from Uploads", use_container_width=True):
            with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                file_tuples = tuple((f.name, f.read()) for f in uploaded_files)
                vs, count   = build_vectorstore_from_bytes(file_tuples)
                if vs:
                    st.session_state.vectorstore   = vs
                    st.session_state.chunk_count   = count
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.success(f"✅ {count:,} chunks indexed!")
                else:
                    st.error("Could not process files. Check they are valid PDFs / DOCX / TXT.")

    st.divider()

    if st.session_state.vectorstore:
        st.markdown(f"""
        <div class="sidebar-section">
          <div class="sidebar-label">Index Stats</div>
          <div class="stat-row"><span class="stat-key">Chunks</span><span class="stat-val">{st.session_state.chunk_count:,}</span></div>
          <div class="stat-row"><span class="stat-key">Files</span><span class="stat-val">{len(st.session_state.indexed_files)}</span></div>
          <div class="stat-row"><span class="stat-key">Model</span><span class="stat-val">{model_name}</span></div>
          <div class="stat-row"><span class="stat-key">Top-K</span><span class="stat-val">{top_k}</span></div>
        </div>""", unsafe_allow_html=True)

        with st.expander("📄 Indexed files"):
            for fname in st.session_state.indexed_files:
                st.markdown(f'<span class="source-tag">{fname}</span>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── MAIN ──────────────────────────────────────────────────────────────────────
is_ready    = bool(st.session_state.vectorstore) and bool(mistral_key)
status_html = (
    '<span class="status-badge status-ready">● Ready</span>' if is_ready else
    '<span class="status-badge status-loading">● Upload documents</span>'
)

st.markdown(f"""
<div class="rag-header">
  <div class="rag-logo">🛡️</div>
  <div>
    <div class="rag-title">Safety Standards Assistant</div>
    <div class="rag-subtitle">Mistral AI · RAG · ChromaDB</div>
  </div>
  <div style="margin-left:auto">{status_html}</div>
</div>""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
      <div class="welcome-icon">📋</div>
      <div class="welcome-title">Ask me about your safety documents</div>
      <div class="welcome-text">
        Upload your safety standard PDFs or Word documents in the sidebar,<br>
        click <strong style="color:#f0a500">Build Index</strong>, then start asking questions.
      </div>
      <div style="margin-top:16px">
        <span class="example-chip">PPE requirements for chemical handling</span>
        <span class="example-chip">Emergency evacuation procedure</span>
        <span class="example-chip">ISO 45001 clause 8.1 summary</span>
        <span class="example-chip">Fire safety inspection checklist</span>
      </div>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
              <div class="msg-avatar avatar-user">👤</div>
              <div class="msg-bubble bubble-user">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            srcs = f'<div class="sources-block"><strong>Sources</strong><br>{msg["sources"]}</div>' if msg.get("sources") else ""
            st.markdown(f"""
            <div class="msg-bot">
              <div class="msg-avatar avatar-bot">🛡️</div>
              <div>
                <div class="msg-bubble bubble-bot">{msg["content"]}</div>
                {srcs}
              </div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
col_input, col_send = st.columns([6, 1])
with col_input:
    user_input = st.text_input(
        "q", label_visibility="collapsed",
        placeholder="e.g. What are the PPE requirements for working at height?",
        key="user_input", disabled=not is_ready,
    )
with col_send:
    send = st.button("Send →", use_container_width=True, disabled=not is_ready)

if (send or user_input) and user_input.strip() and is_ready:
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": question})

    chain = build_chain(st.session_state.vectorstore, model_name, mistral_key, temperature, top_k)

    with st.spinner("Searching documents and generating answer..."):
        t0      = time.time()
        result  = chain.invoke({"question": question})
        elapsed = time.time() - t0

    answer       = result["result"].strip()
    sources_html = format_sources(result.get("source_documents", []))

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "sources": sources_html, "time": f"{elapsed:.1f}s",
    })
    st.rerun()

elif (send or user_input) and not mistral_key:
    st.warning("⚠️ Please enter your Mistral AI API key in the sidebar.")
elif (send or user_input) and not st.session_state.vectorstore:
    st.warning("⚠️ Please upload documents and click **Build Index** first.")
