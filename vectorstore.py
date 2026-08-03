"""
Hybrid retrieval: Chroma for dense vector search, rank_bm25 for sparse
keyword search, fused with Reciprocal Rank Fusion (RRF). Self-hosted/embedded
Chroma has no built-in hybrid search (unlike Qdrant) - this module is the
explicit, hand-rolled equivalent of what Qdrant would do server-side.

Persistence: chromadb.PersistentClient (NOT the deprecated
Settings(persist_directory=...) pattern the original repo used, which
doesn't reliably persist on modern Chroma versions).
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from config import settings
from embeddings import embed_query, embed_texts
from parsing import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    metadata: dict
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


class HybridStore:
    def __init__(self) -> None:
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25: BM25Okapi | None = None
        self._bm25_chunk_ids: list[str] = []
        self._bm25_corpus_tokens: list[list[str]] = []
        self._load_bm25()

    # ---------- indexing ----------

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = embed_texts([c.content for c in chunks])
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.content for c in chunks],
            metadatas=[
                {
                    "page_id": c.page_id,
                    "title": c.title,
                    "version": c.version,
                    "url": c.url,
                    "breadcrumb": c.breadcrumb,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                }
                for c in chunks
            ],
        )
        self._rebuild_bm25_from_collection()
        self._save_bm25()

    def delete_page(self, page_id: str) -> None:
        """Remove all chunks for a page before re-adding its current version -
        keeps stale chunks from a deleted/shrunk page from lingering forever."""
        self._collection.delete(where={"page_id": page_id})
        self._rebuild_bm25_from_collection()
        self._save_bm25()

    def _rebuild_bm25_from_collection(self) -> None:
        data = self._collection.get(include=["documents"])
        self._bm25_chunk_ids = data["ids"]
        self._bm25_corpus_tokens = [_tokenize(doc) for doc in data["documents"]]
        self._bm25 = BM25Okapi(self._bm25_corpus_tokens) if self._bm25_corpus_tokens else None

    def _save_bm25(self) -> None:
        with open(settings.bm25_index_path, "wb") as f:
            pickle.dump(
                {"chunk_ids": self._bm25_chunk_ids, "corpus_tokens": self._bm25_corpus_tokens}, f
            )

    def _load_bm25(self) -> None:
        path = Path(settings.bm25_index_path)
        if not path.exists():
            self._rebuild_bm25_from_collection()
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25_chunk_ids = data["chunk_ids"]
        self._bm25_corpus_tokens = data["corpus_tokens"]
        self._bm25 = BM25Okapi(self._bm25_corpus_tokens) if self._bm25_corpus_tokens else None

    # ---------- retrieval ----------

    def hybrid_search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        dense_results = self._dense_search(query, top_k)
        sparse_results = self._sparse_search(query, top_k)
        return self._fuse(dense_results, sparse_results)

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, str, dict]]:
        if self._collection.count() == 0:
            return []
        query_vec = embed_query(query)
        res = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return list(zip(ids, docs, metas))

    def _sparse_search(self, query: str, top_k: int) -> list[tuple[str, str, dict]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        if not ranked_idx:
            return []
        chunk_ids = [self._bm25_chunk_ids[i] for i in ranked_idx]
        got = self._collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        by_id = {cid: (doc, meta) for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])}
        return [(cid, *by_id[cid]) for cid in chunk_ids if cid in by_id]

    def _fuse(
        self,
        dense: list[tuple[str, str, dict]],
        sparse: list[tuple[str, str, dict]],
        k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across the
        lists a chunk appears in. Parameter-free in practice (k=60 is the
        standard default from the original RRF paper) - no hand-tuned
        weighting between dense and sparse scores to get wrong."""
        k = k or settings.rrf_k
        by_id: dict[str, RetrievedChunk] = {}

        for rank, (cid, doc, meta) in enumerate(dense):
            rc = by_id.setdefault(cid, RetrievedChunk(chunk_id=cid, content=doc, metadata=meta))
            rc.dense_rank = rank
            rc.rrf_score += 1.0 / (k + rank + 1)

        for rank, (cid, doc, meta) in enumerate(sparse):
            rc = by_id.setdefault(cid, RetrievedChunk(chunk_id=cid, content=doc, metadata=meta))
            rc.sparse_rank = rank
            rc.rrf_score += 1.0 / (k + rank + 1)

        return sorted(by_id.values(), key=lambda r: r.rrf_score, reverse=True)


_store: HybridStore | None = None


def get_store() -> HybridStore:
    global _store
    if _store is None:
        _store = HybridStore()
    return _store
