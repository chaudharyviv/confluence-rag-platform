"""
LangGraph orchestration:

START -> router
router --in_domain-->      retrieve -> rerank -> generate -> groundedness -> log -> END
router --needs_external--> external_search -> log -> END
router --out_of_domain-->  refuse -> log -> END
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
    router_confidence: float
    candidates: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    answer: str
    groundedness_verdict: str
    unsupported_claims: list[str]
    source_type: str
    start_time: float


def router_node(state: RAGState) -> RAGState:
    decision = llm.classify_domain(state["question"])
    return {
        "router_decision": decision.label,
        "router_confidence": decision.confidence,
    }


def retrieve_node(state: RAGState) -> RAGState:
    store = get_store()
    candidates = store.hybrid_search(state["question"], top_k=settings.retrieve_top_k)
    return {"candidates": candidates}


def rerank_node(state: RAGState) -> RAGState:
    reranked = rerank_chunks(state["question"], state["candidates"], top_k=settings.rerank_top_k)
    return {"reranked": reranked}


def generate_node(state: RAGState) -> RAGState:
    answer = llm.generate_answer(state["question"], state["reranked"])
    return {"answer": answer, "source_type": "internal"}


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
        router_decision=state["router_decision"],
        router_confidence=state.get("router_confidence"),
        chunk_ids=[c.chunk_id for c in state.get("reranked", [])],
        groundedness_verdict=state.get("groundedness_verdict"),
        answer=state["answer"],
        source_type=state["source_type"],
        latency_ms=elapsed_ms,
    )
    return {}


def _route_after_classification(state: RAGState) -> str:
    return state["router_decision"]  # "in_domain" | "needs_external" | "out_of_domain"


def build_graph():
    g = StateGraph(RAGState)

    g.add_node("router", router_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("generate", generate_node)
    g.add_node("groundedness", groundedness_node)
    g.add_node("external_search", external_search_node)
    g.add_node("refuse", refuse_node)
    g.add_node("log", log_node)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        _route_after_classification,
        {
            "in_domain": "retrieve",
            "needs_external": "external_search",
            "out_of_domain": "refuse",
        },
    )
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
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
