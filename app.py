"""
Streamlit is the whole interface for v1. Same `streamlit run app.py` works
on a laptop, in a Docker container on any host, or on Streamlit Community
Cloud - nothing here is platform-specific.
"""
import contextlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path

# Silence noisy transformers module watcher warnings
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st

# Bridge st.secrets to os.environ for deployment platforms
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
    "How does Claude Sonnet 5 compare to GPT-5.6 on cost and benchmarks?",
]

@st.cache_data
def load_golden_lookup() -> dict[str, dict]:
    """Golden QA set, keyed by exact question text."""
    path = Path(__file__).parent / settings.golden_set_path
    if not path.exists():
        return {}
    lookup: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lookup[row["question"]] = row
    return lookup

def _precision_recall(retrieved_ids: list[str], required_sources: list[str]) -> tuple[float, float]:
    """Precision@k and Recall@k calculation against ground truth sources."""
    if not retrieved_ids:
        return 0.0, 0.0
    required = set(required_sources)
    hit = len(required & set(retrieved_ids))
    precision = hit / len(retrieved_ids)
    recall = hit / len(required) if required else 0.0
    return precision, recall


def ragas_available() -> bool:
    """Check availability of Ragas evaluation package."""
    return importlib.util.find_spec("ragas") is not None


def run_eval_live(golden_set_path: str) -> tuple[bool, str]:
    """Run in-process eval gate script capturing stdout."""
    import eval.run_eval as run_eval_module

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            passed = run_eval_module.run(golden_set_path)
    except SystemExit:
        passed = False
    return passed, buf.getvalue()


def render_meta(source_type: str, groundedness_verdict: str | None) -> None:
    """Render source-type and groundedness badges inside a structured row."""
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


def _chunk_meta(c) -> dict:
    meta = getattr(c, "metadata", None) or (c if isinstance(c, dict) else {})
    return meta if isinstance(meta, dict) else {}


def _candidate_row(c) -> dict:
    return {
        "title": _chunk_meta(c).get("breadcrumb") or _chunk_meta(c).get("title", "—"),
        "dense_rank": c.dense_rank if c.dense_rank is not None else 0,
        "sparse_rank": c.sparse_rank if c.sparse_rank is not None else 0,
        "rrf_score": round(c.rrf_score, 4) if getattr(c, "rrf_score", None) is not None else 0.0,
    }


def render_pipeline(
    question: str,
    candidates: list,
    reranked: list,
    router_decision: str | None,
    router_confidence: float | None,
) -> None:
    """Show the retrieval -> fusion -> rerank -> routing breakdown."""
    with st.expander("How this answer was built", icon=":material/route:", expanded=False):
        st.markdown("### 1. Retrieval — two independent searches")
        st.caption(
            "Dense search (Chroma) embeds the query for semantic matching. "
            "Sparse search (BM25) tracks exact keyword overlap. Both run on every query."
        )

        dense_only = sorted(
            (c for c in candidates if getattr(c, "dense_rank", None) is not None), key=lambda c: c.dense_rank
        )[:10]
        sparse_only = sorted(
            (c for c in candidates if getattr(c, "sparse_rank", None) is not None), key=lambda c: c.sparse_rank
        )[:10]

        col_d, col_s = st.columns(2)
        with col_d:
            st.markdown("**Chroma (dense) top hits**")
            if dense_only:
                st.dataframe(
                    [
                        {
                            "title": _chunk_meta(c).get("breadcrumb") or _chunk_meta(c).get("title", "—"),
                            "dense_rank": c.dense_rank,
                        }
                        for c in dense_only
                    ],
                    column_config={"dense_rank": st.column_config.NumberColumn("Rank", format="%d")},
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No dense hits.")
        with col_s:
            st.markdown("**BM25 (sparse) top hits**")
            if sparse_only:
                st.dataframe(
                    [
                        {
                            "title": _chunk_meta(c).get("breadcrumb") or _chunk_meta(c).get("title", "—"),
                            "sparse_rank": c.sparse_rank,
                        }
                        for c in sparse_only
                    ],
                    column_config={"sparse_rank": st.column_config.NumberColumn("Rank", format="%d")},
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No sparse hits.")

        st.markdown("### 2. Fusion — Reciprocal Rank Fusion")
        st.caption(
            f"Combines both rankings via `score = Σ 1 / (k + rank)` (k={settings.rrf_k}). "
            f"{len(candidates)} unique candidates merged."
        )
        if candidates:
            top_candidates = sorted(candidates, key=lambda c: getattr(c, "rrf_score", 0.0), reverse=True)[:10]
            st.dataframe(
                [_candidate_row(c) for c in top_candidates],
                column_config={
                    "rrf_score": st.column_config.ProgressColumn(
                        "RRF Score", format="%.4f", min_value=0.0, max_value=0.1
                    ),
                    "dense_rank": st.column_config.NumberColumn("Dense Rank", format="%d"),
                    "sparse_rank": st.column_config.NumberColumn("Sparse Rank", format="%d"),
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No candidates retrieved.")

        st.markdown("### 3. Rerank — cross-encoder rescoring")
        st.caption(
            f"Jointly rescores (question, chunk) pairs. Top {settings.rerank_top_k} candidates survive."
        )
        if reranked:
            st.dataframe(
                [
                    {
                        "title": _chunk_meta(c).get("breadcrumb") or _chunk_meta(c).get("title", "—"),
                        "section": _chunk_meta(c).get("section", "—"),
                        "tokens": _chunk_meta(c).get("token_count", 0),
                        "rerank_score": c.rerank_score if getattr(c, "rerank_score", None) is not None else 0.0,
                        "passes_threshold": (
                            c.rerank_score > settings.rerank_relevance_threshold
                            if getattr(c, "rerank_score", None) is not None
                            else False
                        ),
                    }
                    for c in reranked
                ],
                column_config={
                    "rerank_score": st.column_config.ProgressColumn(
                        "Relevance Score", format="%.3f", min_value=0.0, max_value=1.0
                    ),
                    "tokens": st.column_config.NumberColumn("Tokens", format="%d"),
                    "passes_threshold": st.column_config.CheckboxColumn("Threshold Met"),
                },
                hide_index=True,
                use_container_width=True,
            )
            with st.popover("Preview chunk text"):
                for i, c in enumerate(reranked):
                    label = _chunk_meta(c).get("breadcrumb") or _chunk_meta(c).get("title", f"Chunk {i}")
                    st.caption(label)
                    st.text(c.content[:500] + ("…" if len(c.content) > 500 else ""))
        else:
            st.caption("No chunks survived reranking.")

        st.markdown("### 4. Precision & recall — against labeled ground truth")
        golden_row = load_golden_lookup().get(question)
        if golden_row and golden_row.get("required_sources"):
            required = golden_row["required_sources"]
            fused_ids = [c.chunk_id for c in candidates]
            reranked_ids = [c.chunk_id for c in reranked]
            p_fused, r_fused = _precision_recall(fused_ids, required)
            p_reranked, r_reranked = _precision_recall(reranked_ids, required)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Precision@{len(fused_ids)} (fused)", f"{p_fused:.2f}")
            m2.metric("Recall (fused)", f"{r_fused:.2f}")
            m3.metric(f"Precision@{len(reranked_ids)} (reranked)", f"{p_reranked:.2f}")
            m4.metric("Recall (reranked)", f"{r_reranked:.2f}")
        else:
            st.caption(
                "No labeled ground truth for this question. Precision and recall "
                "require ground-truth entries in `eval/golden_set.jsonl`."
            )

        st.markdown("### 5. Routing")
        top_score = reranked[0].rerank_score if reranked and getattr(reranked[0], "rerank_score", None) is not None else None
        if router_decision in (None, "retrieval_confirmed") and top_score is not None:
            st.caption(
                f"Top rerank score ({top_score:.3f}) cleared the "
                f"{settings.rerank_relevance_threshold} threshold — answered directly from retrieval."
            )
        else:
            conf = f"{router_confidence:.2f}" if router_confidence is not None else "n/a"
            st.caption(
                f"Router (`{settings.claude_router_model}`) evaluated query — "
                f"decision: `{router_decision}` (confidence {conf})."
            )


def render_sources(reranked: list) -> None:
    """Render expandable list of sources used for the answer."""
    count = len(reranked) if reranked else 0
    with st.expander(f"Sources used ({count})", icon=":material/link:", expanded=False):
        if not reranked:
            st.caption("No internal sources used for this answer.")
            return
        for c in reranked:
            meta = getattr(c, "metadata", None) or (c if isinstance(c, dict) else {})
            if not isinstance(meta, dict):
                meta = {}
            title = meta.get("breadcrumb") or meta.get("title") or "Untitled source"
            url = meta.get("url") or "#"
            st.markdown(f"- **{title}** ([source]({url}))")


def render_empty_state() -> None:
    """Show empty state using clean selection pills for example prompts."""
    st.info(
        "Ask anything about the knowledge base. "
        "Answers are grounded in internal documents whenever possible.",
        icon=":material/lightbulb:",
    )
    golden = load_golden_lookup()
    examples = list(golden.keys()) if golden else EXAMPLE_QUESTIONS
    
    selected_example = st.pills("Try an example question:", examples, selection_mode="single")
    if selected_example:
        st.session_state._pending_question = selected_example
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

    with st.expander("Eval history (Ragas)", expanded=False):
        golden_set_path = str(Path(__file__).parent / settings.golden_set_path)
        if ragas_available():
            if st.button(
                "Run eval now",
                icon=":material/play_arrow:",
                use_container_width=True,
                help="Runs evaluation pipeline against golden dataset.",
            ):
                with st.spinner("Running evaluation suite..."):
                    passed, log = run_eval_live(golden_set_path)
                if passed:
                    st.success("Release gate: PASSED", icon=":material/check_circle:")
                else:
                    st.error("Release gate: FAILED", icon=":material/cancel:")
                with st.expander("Eval run log", expanded=False):
                    st.code(log, language="text")
        else:
            st.caption(
                "Live eval needs additional packages: "
                "`pip install -r eval/requirements-eval.txt`."
            )

        runs = db.get_recent_eval_runs(limit=20)
        if not runs:
            st.caption("No eval runs recorded yet.")
        else:
            runs = list(reversed(runs))
            st.line_chart(
                {
                    "run_ts": [r.run_ts for r in runs],
                    "faithfulness": [r.faithfulness for r in runs],
                    "context_recall": [r.context_recall for r in runs],
                    "context_precision": [r.context_precision for r in runs],
                    "answer_relevancy": [r.answer_relevancy for r in runs],
                },
                x="run_ts",
            )
            st.dataframe(
                [
                    {
                        "run_ts": r.run_ts,
                        "git_sha": (r.git_sha or "—")[:8],
                        "n": r.num_questions,
                        "faithfulness": r.faithfulness,
                        "context_recall": r.context_recall,
                        "context_precision": r.context_precision,
                        "answer_relevancy": r.answer_relevancy,
                    }
                    for r in reversed(runs)
                ],
                hide_index=True,
                use_container_width=True,
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
            with st.container(border=True):
                render_meta(*turn["meta"])
        if "sources" in turn:
            render_pipeline(
                turn.get("question", ""),
                turn.get("candidates", []),
                turn["sources"],
                turn.get("router_decision"),
                turn.get("router_confidence"),
            )
            render_sources(turn["sources"])

typed_question = st.chat_input("Ask a question about the knowledge base...")
question = st.session_state.pop("_pending_question", None) or typed_question

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        try:
            with st.status("Processing query pipeline...", expanded=True) as status:
                st.write("Performing hybrid retrieval & fusion...")
                result = ask(question)
                st.write("Reranking results and confirming groundedness...")
                status.update(label="Response generated", state="complete", expanded=False)

            st.markdown(result["answer"])
            
            with st.container(border=True):
                render_meta(result["source_type"], result.get("groundedness_verdict"))

            reranked = result.get("reranked") or []
            render_pipeline(
                question,
                result.get("candidates") or [],
                reranked,
                result.get("router_decision"),
                result.get("router_confidence"),
            )
            render_sources(reranked)

            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "meta": (result["source_type"], result.get("groundedness_verdict")),
                    "question": question,
                    "sources": reranked,
                    "candidates": result.get("candidates") or [],
                    "router_decision": result.get("router_decision"),
                    "router_confidence": result.get("router_confidence"),
                }
            )
        except Exception as exc:
            st.error(
                "Something went wrong while generating the answer. "
                "Please try again or rephrase your question.",
                icon=":material/error:",
            )
            st.caption(f"Details: {type(exc).__name__}")