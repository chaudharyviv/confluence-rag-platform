"""
Reranking via a small self-hosted HuggingFace cross-encoder - no external API
call, no added per-query cost. Deliberately using the small MiniLM variant
(~22M params) rather than something like bge-reranker-v2-m3 (568M params):
on a memory-constrained free host (e.g. Streamlit Community Cloud's ~1GB
ceiling), a large reranker competes for RAM with everything else in the same
process. If eval numbers ever show this model is the retrieval bottleneck,
that's a deliberate trade-off to revisit - not a default to assume is fine
forever.
"""
from sentence_transformers import CrossEncoder

from config import settings
from vectorstore import RetrievedChunk

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(settings.reranker_model)
    return _model


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
    # top_k here is just this function's fallback; callers pass
    # settings.rerank_top_k explicitly (see graph.py's rerank_node).
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query, c.content) for c in candidates]
    scores = model.predict(pairs)
    for c, score in zip(candidates, scores):
        c.rerank_score = float(score)
    return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)[:top_k]
