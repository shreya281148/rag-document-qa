"""Streamlit chat interface for rag-document-qa.

Run:  streamlit run app.py

Upload PDFs / text files, index them, then ask questions. Every answer shows the exact
passages (file, page, similarity) it was based on. Everything runs on this computer.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import streamlit as st

from ragqa import RAGPipeline, load_config
from ragqa.doctor import check_setup
from ragqa.errors import RagError

SAMPLE_DIR = Path(__file__).parent / "sample_docs"

st.set_page_config(page_title="Local Document Q&A", layout="wide")


# --------------------------------------------------------------------------- helpers
def make_config(offline: bool, model: str, top_k: int, chunk_size: int, chunk_overlap: int):
    cfg = load_config()
    if offline:
        cfg.embeddings.provider, cfg.llm.provider = "hash", "extractive"
    else:
        cfg.llm.model = model
    cfg.retrieval.top_k = top_k
    cfg.chunking.chunk_size, cfg.chunking.chunk_overlap = chunk_size, chunk_overlap
    cfg.validate()
    return cfg


@st.cache_resource(show_spinner=False)
def get_pipeline(offline: bool, model: str, top_k: int, chunk_size: int, chunk_overlap: int, persist_dir: str):
    # Cached so the vector database is opened once, not on every click. `persist_dir` is part of
    # the cache key on purpose: a different folder must give a different pipeline.
    return RAGPipeline(make_config(offline, model, top_k, chunk_size, chunk_overlap))


@st.cache_data(ttl=15, show_spinner=False)
def run_checks(offline: bool, model: str, llm_url: str, emb_url: str):
    cfg = make_config(offline, model, 4, 800, 120)
    return [(c.name, c.ok, c.detail) for c in check_setup(cfg, timeout=2)]


# --------------------------------------------------------------------------- sidebar
base = load_config()
st.sidebar.title("Local Document Q&A")
st.sidebar.caption("Retrieval-Augmented Generation, running on this computer.")

st.sidebar.subheader("Settings")
offline = st.sidebar.toggle(
    "Offline demo mode (no Ollama)",
    value=base.llm.provider == "extractive" and base.embeddings.provider == "hash",
    help="Uses a tiny built-in embedder and quotes sentences instead of using a language model. "
    "Good for trying the app without installing anything; much weaker than a real model.",
)
model = st.sidebar.text_input("Language model (Ollama)", value=base.llm.model, disabled=offline)
top_k = st.sidebar.slider("Passages per answer (top-k)", 1, 10, base.retrieval.top_k)
chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 2000, base.chunking.chunk_size, step=100)
chunk_overlap = st.sidebar.slider("Chunk overlap (characters)", 0, 400, min(base.chunking.chunk_overlap, 400), step=20)
st.sidebar.caption("Chunk settings apply to documents indexed from now on. Indexes are kept separately per embedding model.")

if chunk_overlap >= chunk_size:
    st.sidebar.error("Overlap must be smaller than the chunk size.")
    st.stop()

with st.sidebar.expander("Setup status", expanded=False):
    for name, ok, detail in run_checks(offline, model, base.llm.base_url, base.embeddings.base_url):
        (st.success if ok else st.error)(f"{name}: {detail}")
    st.caption(f"Model server: {'none (offline mode)' if offline else base.llm.base_url}")

pipe = get_pipeline(offline, model, top_k, chunk_size, chunk_overlap, base.storage.persist_dir)

# --------------------------------------------------------------------------- documents
st.sidebar.subheader("Documents")
uploads = st.sidebar.file_uploader("Upload PDF, TXT or MD files", type=["pdf", "txt", "md"], accept_multiple_files=True)
col_a, col_b = st.sidebar.columns(2)

if col_a.button("Index uploads", disabled=not uploads, use_container_width=True):
    try:
        with st.spinner("Reading, chunking and embedding..."), tempfile.TemporaryDirectory() as tmp:
            paths = []
            for f in uploads:
                target = Path(tmp) / Path(f.name).name  # keep the real file name for citations
                target.write_bytes(f.getbuffer())
                paths.append(target)
            report = pipe.ingest(paths)
        st.sidebar.success(f"Indexed {report.files} file(s) into {report.chunks} chunks.")
    except RagError as exc:
        st.sidebar.error(str(exc))

if col_b.button("Load samples", use_container_width=True, help="Index the example documents in sample_docs/"):
    try:
        with st.spinner("Indexing sample documents..."):
            report = pipe.ingest([SAMPLE_DIR])
        st.sidebar.success(f"Indexed {report.files} sample file(s) into {report.chunks} chunks.")
    except RagError as exc:
        st.sidebar.error(str(exc))

indexed = pipe.indexed_sources()
if indexed:
    for name, n in indexed.items():
        st.sidebar.markdown(f"- `{name}` ({n} chunks)")
    if st.sidebar.button("Clear index", use_container_width=True):
        pipe.reset()
        st.session_state.pop("messages", None)
        st.rerun()
else:
    st.sidebar.info("No documents indexed yet.")

# --------------------------------------------------------------------------- chat
st.header("Ask your documents")
st.session_state.setdefault("messages", [])


def cited_numbers(answer: str) -> set[int]:
    """Which [n] markers does the answer actually contain?"""
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer or "")}


def show_sources(hits, answer: str = "") -> None:
    if not hits:
        return
    cited = cited_numbers(answer)
    with st.expander(f"Sources ({len(hits)}; {len(cited & set(range(1, len(hits) + 1)))} cited in the answer)"):
        st.caption("Passages given to the model, most similar first. 'Cited' means the answer refers to it.")
        for n, h in enumerate(hits, start=1):
            tag = " - **cited**" if n in cited else ""
            st.markdown(f"**[{n}]** `{h.source}`, page {h.page} - similarity {h.score:.2f}{tag}")
            st.text(h.text)  # plain text: document content is never rendered as markdown/HTML


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources"), msg["content"])

if not indexed:
    st.info("Upload documents (or click **Load samples**) in the sidebar to get started.")

if question := st.chat_input("Ask a question about your documents", disabled=not indexed):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            hits, stream = pipe.answer_stream(question)
            answer = st.write_stream(stream)
            show_sources(hits, answer)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": hits})
        except RagError as exc:
            st.error(str(exc))
