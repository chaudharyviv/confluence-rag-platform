"""
LangGraph orchestration - retrieval-informed routing:

START -> retrieve -> rerank -> route_after_rerank (function, no LLM call)
    top rerank score > threshold --> generate -> groundedness -> log -> END
    weak/no match               --> router (Claude call) -> route_after_router
        needs_external --> external_search -> log -> END
        out_of_domain  --> refuse -> log -> END

This replaced an earlier version where a Claude router classified the
question BEFORE retrieval ever ran - which meant "current flagship"-style
phrasing could get routed external even when the knowledge base had an
exact, current answer sitting in the index. Retrieval now runs first and is
the primary signal; the LLM router is only consulted as a fallback classifier
when retrieval itself comes up weak (empty, or the top match doesn't look
relevant) - at that point the real question is "is this out of scope, or
does it need live info," which retrieval strength can't answer by itself.
"""
from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

import db
import llm
from config import settings
from reranker import rerank as rerank_chunks
from vectorstore import RetrievedChunk, get_store


class RAGState(TypedDict, total=False):
    question: str
    router_decision: str
    router_confidence: float | None
    candidates: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    answer: str
    groundedness_verdict: str
    unsupported_claims: list[str]
    source_type: str
    start_time: float


def retrieve_node(state: RAGState) -> RAGState:
    store = get_store()
    candidates = store.hybrid_search(state["question"], top_k=settings.retrieve_top_k)
    return {"candidates": candidates}


def rerank_node(state: RAGState) -> RAGState:
    reranked = rerank_chunks(state["question"], state["candidates"], top_k=settings.rerank_top_k)
    return {"reranked": reranked}


def _top_score(state: RAGState) -> float | None:
    reranked = state.get("reranked") or []
    if not reranked or reranked[0].rerank_score is None:
        return None
    return reranked[0].rerank_score


def route_after_rerank(state: RAGState) -> str:
    """No LLM call here - a fast, free decision based on what retrieval
    actually found. Only falls through to the (slower, costs a Claude call)
    router when the top match is weak or missing."""
    score = _top_score(state)
    if score is not None and score > settings.rerank_relevance_threshold:
        return "retrieved"
    return "weak_or_empty"


def router_node(state: RAGState) -> RAGState:
    """Fallback classifier, only reached when retrieval didn't find a
    confident match. Distinguishes 'needs current/live info' from
    'genuinely out of scope' - a distinction retrieval strength alone can't
    make (a bad search doesn't tell you *why* nothing relevant came back)."""
    decision = llm.classify_domain(state["question"])
    return {
        "router_decision": decision.label,
        "router_confidence": decision.confidence,
    }


def route_after_router(state: RAGState) -> str:
    return state["router_decision"]  # "needs_external" | "out_of_domain" | "in_domain"


def generate_node(state: RAGState) -> RAGState:
    # Reached either via a confident retrieval match (router_decision not yet
    # set) or via the router affirming in_domain despite a weak match -
    # label both cases explicitly for the audit log.
    router_decision = state.get("router_decision") or "retrieval_confirmed"
    answer = llm.generate_answer(state["question"], state["reranked"])
    return {"answer": answer, "source_type": "internal", "router_decision": router_decision}


def groundedness_node(state: RAGState) -> RAGState:
    result = llm.check_groundedness(state["answer"], state["reranked"])
    answer = state["answer"]
    if result.verdict == "unsupported":
        answer = (
            "I couldn't verify this answer against the knowledge base with enough "
            "confidence to present it as grounded. Here's what I found, marked as "
            f"unverified:\n\n{answer}"
        )
    elif result.verdict == "partial" and result.unsupported_claims:
        claims = "; ".join(result.unsupported_claims)
        answer = f"{answer}\n\n*Note: the following wasn't fully supported by the sources: {claims}*"
    return {
        "answer": answer,
        "groundedness_verdict": result.verdict,
        "unsupported_claims": result.unsupported_claims,
    }


def external_search_node(state: RAGState) -> RAGState:
    answer = llm.generate_external_answer(state["question"])
    return {
        "answer": f"*This question needs current information beyond the curated knowledge base:*\n\n{answer}",
        "source_type": "external",
        "reranked": [],
        "groundedness_verdict": "n/a",
    }


def refuse_node(state: RAGState) -> RAGState:
    return {
        "answer": (
            f"This doesn't look like a question about {settings.domain_description}. "
            "Try rephrasing, or ask something within that scope."
        ),
        "source_type": "refused",
        "reranked": [],
        "groundedness_verdict": "n/a",
    }


def log_node(state: RAGState) -> RAGState:
    elapsed_ms = int((time.time() - state.get("start_time", time.time())) * 1000)
    db.log_query(
        question=state["question"],
        router_decision=state.get("router_decision", "retrieval_confirmed"),
        router_confidence=state.get("router_confidence"),
        chunk_ids=[c.chunk_id for c in state.get("reranked", [])],
        groundedness_verdict=state.get("groundedness_verdict"),
        answer=state["answer"],
        source_type=state["source_type"],
        latency_ms=elapsed_ms,
    )
    return {}


def build_graph():
    g = StateGraph(RAGState)

    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("router", router_node)
    g.add_node("generate", generate_node)
    g.add_node("groundedness", groundedness_node)
    g.add_node("external_search", external_search_node)
    g.add_node("refuse", refuse_node)
    g.add_node("log", log_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_conditional_edges(
        "rerank",
        route_after_rerank,
        {
            "retrieved": "generate",
            "weak_or_empty": "router",
        },
    )
    g.add_conditional_edges(
        "router",
        route_after_router,
        {
            "in_domain": "generate",  # router still thinks it's answerable despite weak retrieval
            "needs_external": "external_search",
            "out_of_domain": "refuse",
        },
    )
    g.add_edge("generate", "groundedness")
    g.add_edge("groundedness", "log")
    g.add_edge("external_search", "log")
    g.add_edge("refuse", "log")
    g.add_edge("log", END)

    return g.compile()


_graph = None


def ask(question: str) -> RAGState:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph.invoke({"question": question, "start_time": time.time()})
