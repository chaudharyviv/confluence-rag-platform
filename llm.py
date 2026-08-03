"""
Everything that reasons runs on Claude: domain routing, answer generation,
and groundedness verification. OpenAI is only used for embeddings (embeddings.py).

Structured outputs (router decision, groundedness verdict) go through Claude's
tool-use mechanism rather than "ask for JSON and hope" - a validated,
typed decision instead of the substring-matching the original repos used
(e.g. `is_engineering_question()`, which just checked if a keyword was `in`
the question).
"""
from __future__ import annotations

from dataclasses import dataclass

import anthropic

from config import settings
from vectorstore import RetrievedChunk

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


# ---------------------------------------------------------------- routing ---

_ROUTER_TOOL = {
    "name": "classify_question",
    "description": "Classify whether a question can be answered from the internal knowledge base.",
    "input_schema": {
        "type": "object",
        "properties": {
            "in_domain": {
                "type": "boolean",
                "description": "True if this question is plausibly answerable from the knowledge base's domain.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in this classification, 0.0 to 1.0.",
            },
            "needs_external": {
                "type": "boolean",
                "description": (
                    "True if answering well requires current/live information "
                    "likely to have changed since the knowledge base was last updated "
                    "(e.g. a very recent release, current pricing, today's news)."
                ),
            },
        },
        "required": ["in_domain", "confidence", "needs_external"],
    },
}


@dataclass
class RouterDecision:
    in_domain: bool
    confidence: float
    needs_external: bool

    @property
    def label(self) -> str:
        if self.needs_external:
            return "needs_external"
        if self.in_domain:
            return "in_domain"
        return "out_of_domain"


def classify_domain(question: str) -> RouterDecision:
    resp = _client.messages.create(
        model=settings.claude_router_model,
        max_tokens=256,
        tools=[_ROUTER_TOOL],
        tool_choice={"type": "tool", "name": "classify_question"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"The knowledge base's domain is: {settings.domain_description}\n\n"
                    f"Classify this question: {question}"
                ),
            }
        ],
    )
    tool_use = next(b for b in resp.content if b.type == "tool_use")
    data = tool_use.input
    return RouterDecision(
        in_domain=bool(data["in_domain"]),
        confidence=float(data["confidence"]),
        needs_external=bool(data["needs_external"]),
    )


# ------------------------------------------------------------- generation ---


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Source: {c.metadata.get('breadcrumb', c.metadata.get('title', 'unknown'))}\n{c.content}"
        )
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    context = _format_context(chunks)
    resp = _client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=(
            "Answer the question using ONLY the numbered sources provided. "
            "Cite sources inline like [1], [2]. If the sources don't contain "
            "enough information to answer confidently, say so explicitly rather "
            "than filling gaps from general knowledge."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Sources:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def generate_external_answer(question: str) -> str:
    """Used when the router decides the internal KB is likely stale for this
    question. Without a configured search API this still asks Claude, but the
    caller (graph.py) labels the result as external/unverified in the UI and
    audit log - it should never be presented as internally grounded."""
    resp = _client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=(
            "Answer from your general knowledge. Be explicit that this is not "
            "sourced from the user's curated knowledge base and may be out of date."
        ),
        messages=[{"role": "user", "content": question}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


# --------------------------------------------------------- groundedness -----

_GROUNDEDNESS_TOOL = {
    "name": "verify_groundedness",
    "description": "Verify whether an answer is supported by the given sources.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["supported", "partial", "unsupported"],
                "description": (
                    "'supported' if every claim traces to a source, 'partial' if some "
                    "claims aren't supported, 'unsupported' if the core claim isn't backed."
                ),
            },
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific claims in the answer that are NOT backed by the sources, if any.",
            },
        },
        "required": ["verdict", "unsupported_claims"],
    },
}


@dataclass
class GroundednessResult:
    verdict: str
    unsupported_claims: list[str]


def check_groundedness(answer: str, chunks: list[RetrievedChunk]) -> GroundednessResult:
    context = _format_context(chunks)
    resp = _client.messages.create(
        model=settings.claude_router_model,
        max_tokens=512,
        tools=[_GROUNDEDNESS_TOOL],
        tool_choice={"type": "tool", "name": "verify_groundedness"},
        messages=[
            {
                "role": "user",
                "content": f"Sources:\n\n{context}\n\nAnswer to verify:\n{answer}",
            }
        ],
    )
    tool_use = next(b for b in resp.content if b.type == "tool_use")
    data = tool_use.input
    return GroundednessResult(
        verdict=data["verdict"],
        unsupported_claims=data.get("unsupported_claims", []),
    )
