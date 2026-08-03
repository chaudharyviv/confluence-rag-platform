"""
Central configuration. Every setting comes from os.environ (via .env locally,
st.secrets on Streamlit Community Cloud, repo secrets in GitHub Actions, or
plain container env vars anywhere else). Nothing in the rest of the codebase
should read os.environ directly or branch on "which platform am I running on" -
that logic lives here, once.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op if there's no .env file (e.g. on Streamlit Cloud / CI)


def _get(key: str, default: str | None = None, required: bool = False) -> str:
    val = os.environ.get(key, default)
    if required and not val:
        raise RuntimeError(
            f"Missing required config: {key}. Set it in .env locally, in "
            f"st.secrets on Streamlit Cloud, or as a repo/CI secret."
        )
    return val


@dataclass(frozen=True)
class Settings:
    # --- LLM providers ---
    openai_api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY", required=True))
    anthropic_api_key: str = field(default_factory=lambda: _get("ANTHROPIC_API_KEY", required=True))
    embedding_model: str = field(default_factory=lambda: _get("EMBEDDING_MODEL", "text-embedding-3-small"))
    claude_model: str = field(default_factory=lambda: _get("CLAUDE_MODEL", "claude-sonnet-4-6"))
    claude_router_model: str = field(default_factory=lambda: _get("CLAUDE_ROUTER_MODEL", "claude-haiku-4-5-20251001"))

    # --- Confluence source ---
    confluence_base_url: str = field(default_factory=lambda: _get("CONFLUENCE_BASE_URL", required=True))
    confluence_email: str = field(default_factory=lambda: _get("CONFLUENCE_EMAIL", required=True))
    confluence_api_token: str = field(default_factory=lambda: _get("CONFLUENCE_API_TOKEN", required=True))
    confluence_space_key: str = field(default_factory=lambda: _get("CONFLUENCE_SPACE_KEY", required=True))

    # --- Domain / routing ---
    domain_description: str = field(default_factory=lambda: _get(
        "DOMAIN_DESCRIPTION",
        "a curated knowledge base",
    ))

    # --- Storage: vectors ---
    storage_mode: str = field(default_factory=lambda: _get("STORAGE_MODE", "local"))  # local | git
    chroma_persist_dir: str = field(default_factory=lambda: _get("CHROMA_PERSIST_DIR", "./chroma_db"))
    chroma_collection: str = field(default_factory=lambda: _get("CHROMA_COLLECTION", "kb_chunks"))
    bm25_index_path: str = field(default_factory=lambda: _get("BM25_INDEX_PATH", "./bm25_index.pkl"))

    # --- Storage: relational (audit / eval / sync state) ---
    database_url: str = field(default_factory=lambda: _get("DATABASE_URL", "sqlite:///./local.db"))

    # --- Retrieval tuning ---
    retrieve_top_k: int = field(default_factory=lambda: int(_get("RETRIEVE_TOP_K", "20")))
    rerank_top_k: int = field(default_factory=lambda: int(_get("RERANK_TOP_K", "5")))
    reranker_model: str = field(default_factory=lambda: _get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    rrf_k: int = field(default_factory=lambda: int(_get("RRF_K", "60")))  # standard RRF constant

    # --- Optional external fallback ---
    serpapi_api_key: str = field(default_factory=lambda: _get("SERPAPI_API_KEY", ""))

    def chunking_ready(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
