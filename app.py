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


@st.cache_data
def load_golden_lookup() -> dict[str, dict]:
    """Golden QA set, keyed by exact question text. Rows with
    `required_sources` give us labeled ground truth for a live
    precision@k / recall@k demo - no Ragas / LLM judge needed for that,
    just set overlap against what retrieval actually returned. Missing
    file is fine (e.g. a deploy without the eval extras) - demo panel
    just won't have ground truth to compare against."""
    path = Path(__file__).parent / "eval" / "golden_set.jsonl"
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
    """precision@k = of what we retrieved, how much was actually relevant.
    recall@k = of what's actually relevant, how much did we retrieve.
    Both are plain set overlap against the golden set's labeled
    required_sources - the same ground truth eval/run_eval.py uses for its
    deterministic source-coverage check, just reported as a fraction here
    instead of a pass/fail."""
    if not retrieved_ids:
        return 0.0, 0.0
    required = set(required_sources)
    hit = len(required & set(retrieved_ids))
    precision = hit / len(retrieved_ids)
    recall = hit / len(required) if required else 0.0
    return precision, recall


def ragas_available() -> bool:
    """Cheap check (no import) so the demo button can be disabled/explained
    instead of crashing the whole Streamlit process - ragas is deliberately
    excluded from the app's own requirements.txt (see eval/requirements-eval.txt
    and CLAUDE.md) so the deployed app doesn't carry that weight, which means
    it may genuinely be absent here."""
    return importlib.util.find_spec("ragas") is not None


def run_eval_live(golden_set_path: str) -> tuple[bool, str]:
    """Run the same eval/run_eval.py release gate used in CI, in-process, so
    a training audience can watch it run against the live index instead of
    just looking at historical numbers. Captures stdout so the same log CI
    would produce shows up in the UI. Only called after ragas_available()
    is confirmed - run_eval.run() itself calls sys.exit(1) on a missing
    ragas import, which would take the whole Streamlit server down."""
    import eval.run_eval as run_eval_module

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            passed = run_eval_module.run(golden_set_path)
    except SystemExit:
        # run_eval.run() itself calls sys.exit(1) if `import ragas` raises
        # (e.g. an installed-but-incompatible ragas/langchain-community pin -
        # find_spec() above only confirms the package is *present*, not that
        # importing it actually works). Left uncaught, that SystemExit would
        # unwind straight out of this Streamlit script run and silently
        # truncate the rest of the page for this rerun.
        passed = False
    return passed, buf.getvalue()


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


def _chunk_meta(c) -> dict:
    meta = getattr(c, "metadata", None) or (c if isinstance(c, dict) else {})
    return meta if isinstance(meta, dict) else {}


def _candidate_row(c) -> dict:
    return {
        "title": _chunk_meta(c).get("breadcrumb") or _chunk_meta(c).get("title", "—"),
        "dense_rank": c.dense_rank,
        "sparse_rank": c.sparse_rank,
        "rrf_score": round(c.rrf_score, 4),
    }


def render_pipeline(
    question: str,
    candidates: list,
    reranked: list,
    router_decision: str | None,
    router_confidence: float | None,
) -> None:
    """Show the retrieval -> fusion -> rerank -> routing stages that produced
    this answer - the mechanics behind the badges, for training/demo use."""
    with st.expander("How this answer was built", icon=":material/route:", expanded=False):
        st.markdown("### 1. Retrieval — two independent searches")
        st.caption(
            "Dense search (Chroma) embeds the query and finds chunks close in "
            "vector space — good at *meaning* even with no shared words. Sparse "
            "search (BM25) is classic keyword overlap — good at exact terms, "
            "IDs, and jargon dense embeddings can blur together. Neither alone "
            "is reliable, so both run on every query."
        )

        dense_only = sorted(
            (c for c in candidates if c.dense_rank is not None), key=lambda c: c.dense_rank
        )[:10]
        sparse_only = sorted(
            (c for c in candidates if c.sparse_rank is not None), key=lambda c: c.sparse_rank
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
                    hide_index=True,
                    width="stretch",
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
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption("No sparse hits.")

        st.markdown("### 2. Fusion — Reciprocal Rank Fusion")
        st.caption(
            f"RRF combines the two rankings without needing to compare raw scores "
            f"(a cosine similarity and a BM25 score aren't on the same scale, so "
            f"blending them directly would be arbitrary). Each chunk gets "
            f"`score = Σ 1 / (k + rank)` summed over every list it appears in "
            f"(k={settings.rrf_k}) — showing up near the top of *either* list, "
            f"or moderately in *both*, pushes a chunk up the fused ranking. "
            f"{len(candidates)} unique candidates came out of this step."
        )
        if candidates:
            top_candidates = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)[:10]
            st.dataframe(
                [_candidate_row(c) for c in top_candidates],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No candidates retrieved.")

        st.markdown("### 3. Rerank — cross-encoder rescoring")
        st.caption(
            "RRF is fast but coarse — it never actually reads the chunk against "
            "the question together. A cross-encoder does: it scores every "
            f"(question, chunk) pair jointly — all {len(candidates)} fused "
            f"candidates get rescored, then only the top {settings.rerank_top_k} "
            "survive. Slower than RRF (it's a real forward pass per pair, not "
            "just arithmetic on ranks), but far more precise about *relevance*, "
            "not just keyword/vector overlap — which is exactly why it can "
            "reorder a chunk that RRF ranked poorly up into the final answer."
        )
        if reranked:
            st.dataframe(
                [
                    {
                        "title": _chunk_meta(c).get("breadcrumb") or _chunk_meta(c).get("title", "—"),
                        "section": _chunk_meta(c).get("section", "—"),
                        "tokens": _chunk_meta(c).get("token_count", "—"),
                        "rerank_score": round(c.rerank_score, 4) if c.rerank_score is not None else None,
                        "passes_threshold": (
                            c.rerank_score > settings.rerank_relevance_threshold
                            if c.rerank_score is not None
                            else None
                        ),
                    }
                    for c in reranked
                ],
                hide_index=True,
                width="stretch",
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
            st.caption(
                "Precision = of the chunks we retrieved, how many were actually "
                "relevant (per the golden set's labeled `required_sources`). "
                "Recall = of the chunks that were actually relevant, how many did "
                "we manage to retrieve. Rerank usually trades a bit of recall "
                "(fewer chunks survive) for a lot more precision."
            )
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
                "No labeled ground truth for this question, so no precision/recall "
                "here — that needs a golden set entry with `required_sources` "
                "(see `eval/golden_set.jsonl`). Try one of the example questions "
                "on a fresh chat, or see the aggregate Ragas-judged numbers "
                "(`context_precision` / `context_recall`) under **Eval history** "
                "in the sidebar."
            )

        st.markdown("### 5. Routing")
        top_score = reranked[0].rerank_score if reranked and reranked[0].rerank_score is not None else None
        if router_decision in (None, "retrieval_confirmed") and top_score is not None:
            st.caption(
                f"Top rerank score ({top_score:.3f}) cleared the "
                f"{settings.rerank_relevance_threshold} threshold — answered directly "
                "from retrieval, no LLM router call needed."
            )
        else:
            conf = f"{router_confidence:.2f}" if router_confidence is not None else "n/a"
            st.caption(
                f"Retrieval was weak/empty, so the Claude router "
                f"(`{settings.claude_router_model}`) was consulted — "
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
            # Support both object-style and dict-style metadata
            meta = getattr(c, "metadata", None) or (c if isinstance(c, dict) else {})
            if not isinstance(meta, dict):
                meta = {}
            title = meta.get("breadcrumb") or meta.get("title") or "Untitled source"
            url = meta.get("url") or "#"
            st.markdown(f"- **{title}** ([source]({url}))")


def render_empty_state() -> None:
    """Show a welcoming empty state with example questions. Prefers golden
    set questions when available - those carry labeled ground truth, so
    clicking one lights up the precision/recall panel in the pipeline
    breakdown below (training-demo value: every example question doubles
    as a working retrieval-quality demo)."""
    st.info(
        "Ask anything about the knowledge base. "
        "Answers are grounded in internal documents whenever possible.",
        icon=":material/lightbulb:",
    )
    golden = load_golden_lookup()
    examples = list(golden.keys()) if golden else EXAMPLE_QUESTIONS
    st.caption("Try one of these:")
    cols = st.columns(len(examples))
    for i, q in enumerate(examples):
        with cols[i]:
            if st.button(q, width="stretch", key=f"example_{i}"):
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

    with st.expander("Eval history (Ragas)", expanded=False):
        golden_set_path = str(Path(__file__).parent / "eval" / "golden_set.jsonl")
        if ragas_available():
            if st.button(
                "Run eval now",
                icon=":material/play_arrow:",
                width="stretch",
                help="Runs eval/run_eval.py's release gate in-process against "
                "the live index and Claude — same checks CI runs on every PR.",
            ):
                with st.spinner(
                    f"Running the eval gate on {golden_set_path} — "
                    "asks the full pipeline all 4 golden questions, then "
                    "scores them (Ragas + deterministic checks)..."
                ):
                    passed, log = run_eval_live(golden_set_path)
                if passed:
                    st.success("Release gate: PASSED", icon=":material/check_circle:")
                else:
                    st.error("Release gate: FAILED", icon=":material/cancel:")
                with st.expander("Eval run log", expanded=False):
                    st.code(log, language="text")
        else:
            st.caption(
                "Live eval needs the eval extras: "
                "`pip install -r eval/requirements-eval.txt`."
            )

        runs = db.get_recent_eval_runs(limit=20)
        if not runs:
            st.caption("No eval runs recorded yet — run `python eval/run_eval.py eval/golden_set.jsonl`.")
        else:
            runs = list(reversed(runs))  # chronological for the chart
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
                    for r in reversed(runs)  # newest first in the table
                ],
                hide_index=True,
                width="stretch",
            )

    st.divider()
    if st.button(
        "Clear conversation",
        icon=":material/delete:",
        width="stretch",
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
            render_pipeline(
                turn.get("question", ""),
                turn.get("candidates", []),
                turn["sources"],
                turn.get("router_decision"),
                turn.get("router_confidence"),
            )
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
            # Keep the user message but do not add a broken assistant turn
            # so the conversation stays clean.