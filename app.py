"""
Streamlit is the whole interface for v1. Same `streamlit run app.py` works
on a laptop, in a Docker container on any host, or on Streamlit Community
Cloud - nothing here is platform-specific.
"""
import logging
import os

# Streamlit's file watcher probes every transformers submodule for hot-reload
# purposes and logs a warning (with traceback) when optional deps like
# torchvision are missing. Harmless, but noisy - silence just this logger.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st

# st.secrets is Streamlit Community Cloud's mechanism for env vars, but it
# does NOT populate os.environ automatically - config.py (and everything
# that imports it) only reads os.environ, so bridge it here, before any of
# the imports below happen. On a laptop or CI, st.secrets is just empty and
# this loop does nothing; .env / real env vars take over as normal.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except FileNotFoundError:
    pass  # no secrets.toml - fine locally, .env handles it instead

import db
from config import settings
from graph import ask

st.set_page_config(
    page_title="Knowledge base assistant",
    page_icon=":material/search:",
    layout="centered",
)

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

EXAMPLE_QUESTIONS = [
    "What is the main purpose of this knowledge base?",
    "Summarize the key policies",
    "How do I get started?",
]


def render_meta(source_type: str, groundedness_verdict: str | None) -> None:
    """Render source-type and groundedness badges side-by-side."""
    cols = st.columns([1, 1], gap="small")
    with cols[0]:
        label, icon, color = SOURCE_BADGE.get(
            source_type, ("Unknown", ":material/help:", "gray")
        )
        st.badge(label, icon=icon, color=color)
    with cols[1]:
        verdict = VERDICT_BADGE.get(groundedness_verdict)
        if verdict:
            label, icon, color = verdict
            st.badge(label, icon=icon, color=color)


def render_sources(reranked: list) -> None:
    """Render expandable list of sources used for the answer."""
    count = len(reranked) if reranked else 0
    with st.expander(f"Sources used ({count})", icon=":material/link:", expanded=False):
        if not reranked:
            st.caption("No internal sources used for this answer.")
            return
        for c in reranked:
            # Support both object-style and dict-style metadata
            meta = getattr(c, "metadata", None) or (c if isinstance(c, dict) else {})
            if not isinstance(meta, dict):
                meta = {}
            title = meta.get("breadcrumb") or meta.get("title") or "Untitled source"
            url = meta.get("url") or "#"
            st.markdown(f"- **{title}** ([source]({url}))")


def render_empty_state() -> None:
    """Show a welcoming empty state with example questions."""
    st.info(
        "Ask anything about the knowledge base. "
        "Answers are grounded in internal documents whenever possible.",
        icon=":material/lightbulb:",
    )
    st.caption("Try one of these:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[i]:
            if st.button(q, use_container_width=True, key=f"example_{i}"):
                st.session_state._pending_question = q
                st.rerun()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("About")
    st.caption(
        "Hybrid retrieval (Chroma dense + BM25 sparse, RRF-fused) → "
        "cross-encoder reranking → Claude generation → Claude groundedness check."
    )

    with st.expander("Technical details", expanded=False):
        st.caption(
            f":material/database: Vector store — "
            f"`{settings.chroma_persist_dir}` ({settings.storage_mode} mode)"
        )
        st.caption(
            f":material/history: Audit DB — "
            f"`{settings.database_url.split('://')[0]}`"
        )

    st.divider()
    if st.button(
        "Clear conversation",
        icon=":material/delete:",
        use_container_width=True,
        type="secondary",
    ):
        st.session_state.history = []
        st.session_state.pop("_pending_question", None)
        st.rerun()


# ── Main conversation ────────────────────────────────────────────────────────
if not st.session_state.history:
    render_empty_state()

for turn in st.session_state.history:
    avatar = ":material/smart_toy:" if turn["role"] == "assistant" else None
    with st.chat_message(turn["role"], avatar=avatar):
        st.markdown(turn["content"])
        if turn.get("meta"):
            render_meta(*turn["meta"])
        if "sources" in turn:
            render_sources(turn["sources"])

# Support clicking an example question. st.chat_input must be called on every
# run (regardless of a pending example) or the widget disappears for that run.
typed_question = st.chat_input("Ask a question about the knowledge base...")
question = st.session_state.pop("_pending_question", None) or typed_question

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        try:
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
        except Exception as exc:
            st.error(
                "Something went wrong while generating the answer. "
                "Please try again or rephrase your question.",
                icon=":material/error:",
            )
            st.caption(f"Details: {type(exc).__name__}")
            # Keep the user message but do not add a broken assistant turn
            # so the conversation stays clean.