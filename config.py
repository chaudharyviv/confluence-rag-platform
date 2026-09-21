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
    # os.environ.get(key, default) only falls back to `default` when the key
    # is fully absent - but GitHub Actions sets an env var to an empty string
    # (not "unset") when you reference ${{ secrets.X }} for a secret that was
    # never added. That empty string would otherwise silently shadow a real
    # default (e.g. DATABASE_URL's sqlite:///./local.db), so `or` treats an
    # empty string the same as missing.
    val = os.environ.get(key) or default
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
    claude_model: str = field(default_factory=lambda: _get("CLAUDE_MODEL", "claude-sonnet-5"))
    claude_router_model: str = field(default_factory=lambda: _get("CLAUDE_ROUTER_MODEL", "claude-haiku-4-5-20251001"))
    # Output cap for the two generation-tier (claude_model / Sonnet) calls -
    # generate_answer and generate_external_answer in llm.py. This is what
    # drives Sonnet spend the most (router/groundedness stay on the cheaper
    # claude_router_model with their own small fixed caps) - tune this down
    # if cost is the concern, since real answers here have run well under
    # 1024 tokens in practice.
    claude_max_tokens: int = field(default_factory=lambda: int(_get("CLAUDE_MAX_TOKENS", "1024")))

    # --- Confluence source ---
    # Not required=True here: only reindex.py needs these, and the deployed
    # app shouldn't have to carry Confluence credentials just to answer
    # queries. ConfluenceClient validates these itself when actually used.
    confluence_base_url: str = field(default_factory=lambda: _get("CONFLUENCE_BASE_URL", ""))
    confluence_email: str = field(default_factory=lambda: _get("CONFLUENCE_EMAIL", ""))
    confluence_api_token: str = field(default_factory=lambda: _get("CONFLUENCE_API_TOKEN", ""))
    confluence_space_key: str = field(default_factory=lambda: _get("CONFLUENCE_SPACE_KEY", ""))

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

    # --- Eval / demo ---
    golden_set_path: str = field(default_factory=lambda: _get("GOLDEN_SET_PATH", "eval/golden_set.jsonl"))

    # --- Retrieval tuning ---
    retrieve_top_k: int = field(default_factory=lambda: int(_get("RETRIEVE_TOP_K", "10")))
    rerank_top_k: int = field(default_factory=lambda: int(_get("RERANK_TOP_K", "5")))
    reranker_model: str = field(default_factory=lambda: _get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    rrf_k: int = field(default_factory=lambda: int(_get("RRF_K", "60")))  # standard RRF constant
    # ms-marco-MiniLM cross-encoders are trained as a regression on MS MARCO;
    # scores aren't a calibrated 0-1 probability, but positive vs negative is
    # a reasonable rough cutoff in practice for "is the top match relevant at
    # all." Tune this against your own eval set if it's letting weak matches
    # through or bypassing genuinely well-covered questions.
    rerank_relevance_threshold: float = field(
        default_factory=lambda: float(_get("RERANK_RELEVANCE_THRESHOLD", "0.0"))
    )

    # --- Optional external fallback: live web search (Tavily) ---
    # Empty key = no web search; the external route then answers from Claude's
    # general knowledge (still labelled external).
    tavily_api_key: str = field(default_factory=lambda: _get("TAVILY_API_KEY", ""))
    web_search_max_results: int = field(default_factory=lambda: int(_get("WEB_SEARCH_MAX_RESULTS", "5")))

    def chunking_ready(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
