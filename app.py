"""
Streamlit is the whole interface for v1. Same `streamlit run app.py` works
on a laptop, in a Docker container on any host, or on Streamlit Community
Cloud - nothing here is platform-specific.
"""
import logging

# Streamlit's file watcher probes every transformers submodule for hot-reload
# purposes and logs a warning (with traceback) when optional deps like
# torchvision are missing. Harmless, but noisy - silence just this logger.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st

import db
from config import settings
from graph import ask

st.set_page_config(page_title="Knowledge base assistant", page_icon=":material/search:", layout="centered")

db.init_db()  # safe to call every startup - creates tables if missing

st.title(":material/search: Knowledge base assistant")
st.caption(f"Domain: {settings.domain_description}")

if "history" not in st.session_state:
    st.session_state.history = []

SOURCE_BADGE = {
    "internal": ("Answered from knowledge base", ":material/menu_book:", "blue"),
    "external": ("Answered from general knowledge — may be out of date", ":material/public:", "orange"),
    "refused": ("Out of scope", ":material/block:", "gray"),
}
VERDICT_BADGE = {
    "supported": ("Fully grounded", ":material/verified:", "green"),
    "partial": ("Partially grounded", ":material/warning:", "orange"),
    "unsupported": ("Could not verify", ":material/error:", "red"),
}


def render_meta(source_type: str, groundedness_verdict: str | None) -> None:
    with st.container(horizontal=True):
        label, icon, color = SOURCE_BADGE.get(source_type, ("Unknown", ":material/help:", "gray"))
        st.badge(label, icon=icon, color=color)
        verdict = VERDICT_BADGE.get(groundedness_verdict)
        if verdict:
            label, icon, color = verdict
            st.badge(label, icon=icon, color=color)


def render_sources(reranked: list) -> None:
    with st.expander(f"Sources used ({len(reranked)})", icon=":material/link:", expanded=False):
        if not reranked:
            st.caption("No internal sources used for this answer.")
        for c in reranked:
            st.markdown(
                f"- **{c.metadata.get('breadcrumb', c.metadata.get('title'))}** "
                f"([source]({c.metadata.get('url', '#')}))"
            )


for turn in st.session_state.history:
    with st.chat_message(turn["role"], avatar=":material/smart_toy:" if turn["role"] == "assistant" else None):
        st.markdown(turn["content"])
        if turn.get("meta"):
            render_meta(*turn["meta"])
        if "sources" in turn:
            render_sources(turn["sources"])

question = st.chat_input("Ask a question...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        with st.spinner("Thinking..."):
            result = ask(question)

        st.markdown(result["answer"])
        render_meta(result["source_type"], result.get("groundedness_verdict"))
        reranked = result.get("reranked") or []
        render_sources(reranked)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "meta": (result["source_type"], result.get("groundedness_verdict")),
            "sources": reranked,
        }
    )

with st.sidebar:
    st.subheader("About")
    st.caption(
        "Hybrid retrieval (Chroma dense + BM25 sparse, RRF-fused) → "
        "cross-encoder reranking → Claude generation → Claude groundedness check."
    )
    st.caption(f":material/database: Vector store — `{settings.chroma_persist_dir}` ({settings.storage_mode} mode)")
    st.caption(f":material/history: Audit DB — `{settings.database_url.split('://')[0]}`")
