# Production Confluence RAG - Architecture Design

**Status:** Implemented. This is the original design doc, kept as the design rationale - the "why" behind each choice. See [`README.md`](README.md) for the system as built (setup, config reference, current architecture diagrams). A few decisions changed during implementation; those are called out inline as **Implementation note** callouts rather than silently edited away.
**Supersedes:** `Confluence-Rag`, `Confluence-Serpapi-GraphRag`
**Fits roadmap:** Aug 2026 depth phase - production hybrid RAG + Ragas eval

---

## 1. Design Principles

1. **Grounding is enforced, not assumed.** An answer without verifiable support from retrieved context does not ship to the user.
2. **Retrieval quality is measured, not guessed.** Ragas eval is a CI gate, not a notebook you run once.
3. **Runs anywhere, same codebase.** Local laptop, GitHub Actions, Streamlit Community Cloud, Render, a VPS - the app doesn't hardcode any one host. Behavior that differs by environment (where the index lives, which database backend) is controlled by config/env vars, not by branching code per platform.
4. **Database gap gets filled honestly.** A real SQLite/Postgres schema for the audit log and eval history - not a local JSON file pretending to be a database, which is what repo1 actually shipped.

---

## 2. High-Level Architecture

```
┌─────────────┐   reindex.py (same script       ┌──────────────────┐
│  Confluence   │   everywhere: manual on a     │  Reindex Job         │
│  (source)     │   laptop, cron on a VPS,      │  (parse+chunk+embed) │
└─────────────┘   or a GitHub Action)          └─────────┬──────────┘
                                                    │ writes
                                                    ▼
                              ┌──────────────────────────────┐
                              │  chroma_db/  +  bm25_index.pkl   │
                              │  (STORAGE_MODE=local: written    │
                              │   directly, disk persists.       │
                              │   STORAGE_MODE=git: committed     │
                              │   back to repo for ephemeral      │
                              │   hosts to pull on redeploy)      │
                              └─────────────────┬─────────────┘
                                                    ▼
┌───────────┐   ┌───────────────────────────────┴───────────────────┐
│  Streamlit  │──▶│         LangGraph Orchestrator (in-process)            │
│  app        │   │  HybridRetrieve(Chroma dense + BM25 sparse, RRF      │
│  (laptop,   │   │  fused in code) → Rerank(small HF cross-encoder)     │
│  Community  │   │  → confident match? → Generate(Claude); weak/empty   │
│  Cloud,     │   │  → Router(Claude) → Generate | ExternalSearch |      │
│  Docker     │   │  Refuse → Groundedness(Claude) → Log → Respond       │
│  anywhere)  │   └───────────────────┬─────────────────────────────────┘
└───────────┘                           │
                                          │ writes audit rows
       │ DATABASE_URL env var                │ (same DB, any environment)
       ▼                                        ▼
┌───────────────────────────────────────────────────┐
│  SQLite by default - a local file, no signup,        │
│  no risk of "did the free tier pause again."          │
│  Swap DATABASE_URL to Neon only if you want           │
│  audit history to survive across deployed sessions   │
│  - query_audit_log · eval_runs · page_versions        │
└───────────────────────────────────────────────────┘
```

Everything host-specific is a config value (`STORAGE_MODE`, `DATABASE_URL`, API keys), never a code fork. The same `streamlit run app.py` works on a laptop, in a Dockerfile on Render/Railway/Fly.io, or on Streamlit Community Cloud.

**Providers, explicitly:** OpenAI (embeddings) · Claude (routing, generation, groundedness) · small HuggingFace cross-encoder, self-hosted in-process (reranking) · Chroma, embedded (vectors) · `rank_bm25`, in-process (sparse retrieval) · SQLite by default, Neon Postgres optional (audit log, eval history).

---

## 3. Layer-by-Layer

### 3.1 Ingestion
- Keep incremental sync via `version.number` - but move the version table into a real database (SQLite by default), not a local JSON file.
- Parse Confluence **storage-format XHTML** properly with BeautifulSoup:
  - Headings → section breadcrumbs (`Title > H2 > H3`), preserved as chunk metadata for citation quality.
  - `<table>` → converted to Markdown tables, not flattened text.
  - Code macros (`<ac:structured-macro ac:name="code">`) → fenced code blocks.
  - This alone fixes a real correctness bug in repo1: config/version tables were being turned into unstructured prose, which is exactly the content most likely to be asked about.
- Add a webhook receiver (Confluence supports page-updated webhooks) so sync isn't purely poll-based long-term. Poll-based is fine for v1.

### 3.2 Chunking
- Section-aware chunking: split on heading boundaries first, then token-aware sliding window *within* a section (so a chunk never straddles two unrelated headings).
- Each chunk carries: `breadcrumb`, `page_id`, `version`, `url`, `chunk_index`, `token_count`.
- Table chunks are never split mid-table.

### 3.3 Embedding + Storage - designed around the free-tier constraint
- **Embeddings:** OpenAI `text-embedding-3-small` (1536-dim).
- **Vector store:** **Chroma**, embedded (`PersistentClient`), use `chromadb.PersistentClient(path="chroma_db")` instead).
- **Portability via one config flag, `STORAGE_MODE`:**
  - `STORAGE_MODE=local` - reindex writes directly to `./chroma_db/`, app reads from the same folder. This is all you need on a laptop, a VPS, or any host with a real persistent volume (Render/Railway paid disk, a Docker volume, your own machine).
  - `STORAGE_MODE=git` - reindex commits `chroma_db/` + `bm25_index.pkl` back to the repo after building them; the live app only reads. This is the mode for platforms with ephemeral disk (Streamlit Community Cloud is the main one) - auto-redeploy on push pulls the fresh index.
  - Same `reindex.py` script either way; the only difference is a `git add/commit/push` at the end when `STORAGE_MODE=git`. No per-platform code paths in the app itself.
- **Audit / eval store - default to SQLite, not a hosted DB:** `DATABASE_URL` env var selects the backend, defaulting to `sqlite:///./local.db`. For a personal portfolio project (not a service with real users depending on uptime), losing audit history when a free-tier app sleeps is a statable trade-off, not a flaw worth adding an external dependency to avoid - "this demo uses SQLite; production would use a durable store" is a legitimate, honest line in the README.
  - **If you do want the deployed app's history to persist across sessions**, use **Neon**, not Supabase: Neon auto-suspends idle compute but resumes automatically on the next query in under a second, no manual step. Supabase's free tier pauses the *entire project* after 7 days of inactivity and requires manually logging in to unpause it - exactly the failure mode you'd hit when someone (an interviewer, future-you) opens the demo after a gap. Same SQLAlchemy code either way, so this is a config change whenever you want it, not a decision to make now.
  - This still functions as your portfolio's first database-backed project either way - SQLite is a real, appropriate choice for this scale, not a placeholder.

### 3.4 Retrieval - Hybrid, built by hand since Chroma doesn't do it natively
- Self-hosted/embedded Chroma has no built-in sparse or hybrid search (Chroma Cloud has recently started advertising hybrid/full-text - worth re-checking at build time, but don't design around an unconfirmed managed feature).
- So: dense retrieval via Chroma (top 20), sparse retrieval via `rank_bm25` (pure Python, no server, index loaded from the same committed artifact) (top 20), fused with **Reciprocal Rank Fusion** in a LangGraph node - same RRF logic Qdrant would've done server-side, just explicit in application code instead of hidden in the DB.
- Rerank fused top-20 down to top 5 with a **small** HuggingFace cross-encoder - `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22M params, CPU-friendly, loads in well under 100MB) rather than `bge-reranker-v2-m3` (568M params - would blow the 1GB memory ceiling alongside everything else running in the same process). If eval numbers later show this smaller model is the retrieval bottleneck, that's a signed-off trade-off to revisit, not a guess.

### 3.5 Routing - LLM-based, not keyword lists
- Replace `is_engineering_question()` and `evidence_is_sufficient()` (the `"how to" → always False` bug) with a Claude router node (Haiku-tier is enough for a classification task) with a **structured output schema** (`in_domain: bool`, `confidence: float`, `needs_external: bool`).
- The domain itself is a config value, not hardcoded logic: `DOMAIN_DESCRIPTION` env var (e.g. *"the global LLM landscape - models, providers, release dates, context windows, licensing, benchmarks"*) feeds the router's classification prompt. Same code works whatever topic the Confluence space actually covers.
- For a fast-moving topic like this one, the `needs_external` path is doing real work, not just demonstrating a pattern: your curated pages will genuinely lag new model releases, so "answer from my notes" vs. "this needs a live check" is a real distinction the router has to get right - worth reflecting that in the eval set (a few golden questions that *should* route external).
- Structured output means you get a typed, validated decision instead of substring matching - and it's auditable (log the classification + confidence per query).

> **Implementation note - router moved to run *after* retrieval, not before.** The original plan below had the Claude router run first and gate whether retrieval happened at all. In practice that misrouted "current flagship"-style phrasing to `needs_external` even when the knowledge base held an exact, current answer - the router was guessing at retrievability without checking it. `graph.py` runs `retrieve → rerank` first, and only calls the router as a fallback classifier when the top reranked score is at or below `RERANK_RELEVANCE_THRESHOLD` (weak or empty match). At that point the real open question - "is this out of scope, or does it need live info?" - is one retrieval strength genuinely can't answer by itself, which is exactly what the router is for. See the `graph.py` module docstring for the full rationale, and §5 below for the as-built state machine (kept for reference; the current diagram lives in `README.md`).

### 3.6 Groundedness Verification (new - neither repo had this)
- After generation, a second LLM pass checks each sentence of the answer against the cited chunks and returns a per-sentence support label.
- Any unsupported sentence → either strip it, or downgrade the whole answer to "partial answer, see sources" rather than presenting it as fully grounded.
- This is the single highest-leverage addition for a banking-adjacent portfolio piece: it's the difference between "a RAG demo" and "a RAG system that knows what it doesn't know."

### 3.7 External Fallback (kept, re-architected)
- The external path stays - it's a legitimately useful feature - but it's now reached only via the router's `needs_external` decision, not a hardcoded "procedural questions always go external" rule.
- External answers are visually/structurally distinguished from internal ones in the UI (already partially done in repo2 - worth keeping).

> **Implementation note - SerpAPI isn't wired to a live call yet.** `SERPAPI_API_KEY` exists as a config value, but `llm.generate_external_answer()` currently answers from Claude's general knowledge, explicitly instructed to flag itself as unsourced/possibly-stale - it does not call SerpAPI. The router → external-path plumbing described above is fully built; only the actual search call is deferred. Wiring a real SerpAPI (or similar) call into that function is a self-contained follow-up, not a design change.

### 3.8 Evaluation (Ragas) - CI gate
- Build a golden set of 30–50 QA pairs from real Confluence content (question, expected answer, expected source page).
- Ragas metrics: `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`.
- Store results in the `eval_runs` table (trend over time, not just pass/fail).
- Wire into GitHub Actions: PRs that touch chunking/retrieval/prompts must clear a faithfulness threshold before merge. This is the concrete deliverable behind "add RAG eval (Ragas)" on your Aug roadmap.

### 3.9 Observability
- Structured tracing per request (LangFuse or OpenTelemetry): router decision, retrieved chunk IDs + scores, rerank scores, groundedness verdict, latency per stage, token cost.
- This is also what makes the eval numbers debuggable instead of a black box.

### 3.10 Serving
- **Streamlit** is the whole interface for v1 - reusing your existing `_QueueHolder` + `st.fragment` async pattern for streamed responses. Same `streamlit run app.py` command whether it's running on your laptop, inside a Dockerfile on Render/Railway/Fly.io, or on Streamlit Community Cloud.
- **Deliberately not Vercel.** Vercel is built for Next.js + short-lived serverless functions; Streamlit needs a long-running Python process holding a WebSocket connection, which the Hobby tier isn't built to run. Moving to Vercel would mean rewriting the UI in Next.js *and* still standing up a separate Python host for the actual RAG pipeline (Chroma, the reranker, LangGraph) - trading one free host for two, for a frontend-framework change that doesn't strengthen the thing this project is meant to demonstrate. Worth revisiting only if the goal ever becomes "show full-stack breadth," which is a different project.
- Ingestion/reindexing is a **separate CLI script** (`reindex.py`), not a web endpoint - run it manually on a laptop, on a cron, or as a GitHub Action. Keeping it decoupled from the web app is also just correct design: a long-running embedding job has no business blocking a Streamlit request thread.
- A REST layer (FastAPI) is easy to add later if something external ever needs to call this programmatically - not built now since there's no consumer for it yet. Better to add it when a real need shows up than to carry unused surface area.

### 3.10a Configuration Summary (what makes this portable)
```
OPENAI_API_KEY, ANTHROPIC_API_KEY        - required everywhere
CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN   - required for reindex.py
CONFLUENCE_SPACE_KEY = <your space>       - e.g. a "World LLMs" space you author
DOMAIN_DESCRIPTION = <short topic desc>   - feeds the router's classification prompt
STORAGE_MODE = local | git                - default: local
CHROMA_PERSIST_DIR = ./chroma_db          - default
DATABASE_URL = sqlite:///./local.db       - default; swap for a Neon URL only if you want audit history to persist across deployed sessions
```
Local dev: `.env` file. Streamlit Community Cloud: `st.secrets` (same variable names). GitHub Actions: repo secrets. Any Docker host: environment variables passed to the container. One `config.py` reads from `os.environ` regardless of source - the app never needs to know which platform it's running on.

### 3.11 Compliance / Audit (banking-relevant, was entirely absent)
- `query_audit_log` table: question, router decision, chunks used, groundedness verdict, answer, timestamp, source classification (internal/external/refused).
- This turns "regulated banking environment" from a bio-line into something the project actually demonstrates.

---

## 4. Data Model

**Chroma collection `kb_chunks`** (name configurable via env var; persisted under `chroma_db/`, committed to Git):
```
id, embedding[1536] (OpenAI text-embedding-3-small),
document: content text,
metadata: { page_id, title, breadcrumb, section, url, version, chunk_index, token_count }
```

**`bm25_index.pkl`** - a `rank_bm25` `BM25Okapi` index over the same chunk set, tokenized content + chunk IDs, rebuilt alongside the Chroma index in the same reindex job.

**SQLite by default** (`local.db`), or **Neon Postgres** if you opt into hosted persistence - same schema either way, via SQLAlchemy:
```
page_versions(page_id, version, synced_at)
query_audit_log(id, question, router_decision, confidence, chunk_ids[], groundedness_verdict, answer, source_type, ts)
eval_runs(id, run_ts, context_precision, context_recall, faithfulness, answer_relevancy, git_sha)
```

---

## 5. LangGraph State Machine

**Original plan (superseded - router-first; kept for historical context):**

```
START → Router
Router --in_domain--> HybridRetrieve
Router --needs_external--> ExternalSearch
Router --out_of_domain--> Refuse

HybridRetrieve(Chroma dense + rank_bm25 sparse, RRF fused in code) → Rerank(small HF cross-encoder) → Generate(Claude) → GroundednessCheck(Claude)
GroundednessCheck --supported--> Log → END
GroundednessCheck --unsupported--> Downgrade → Log → END

ExternalSearch → Generate(external) → Log → END
```

**As built (`graph.py`) - retrieval-first, router as fallback classifier:**

```
START → HybridRetrieve(Chroma dense + rank_bm25 sparse, RRF fused in code) → Rerank(small HF cross-encoder)

Rerank --top score > RERANK_RELEVANCE_THRESHOLD--> Generate(Claude)
Rerank --weak or empty match--> Router(Claude, structured output)

Router --in_domain--> Generate(Claude)        # router still thinks it's answerable despite weak retrieval
Router --needs_external--> ExternalSearch
Router --out_of_domain--> Refuse

Generate(Claude) → GroundednessCheck(Claude)
GroundednessCheck --supported--> Log → END
GroundednessCheck --partial/unsupported--> Downgrade answer text → Log → END

ExternalSearch → Log → END
Refuse → Log → END
```

Why the change: gating retrieval behind the router meant the router had to guess whether the knowledge base could answer a question *before any search ran* - which misrouted phrasing like "current flagship" externally even when the index had an exact, current answer sitting in it. Running retrieval first makes "did we actually find something good?" the primary signal, and reserves the (slower, Claude-call) router for the genuinely ambiguous case: a weak or empty match, where the open question is *why* nothing relevant came back, not whether it did.

---

## 6. Repo / Naming Decision (open)

Two naive repos get retired. Suggested name for the new one, pick one:
- `confluence-rag-platform`
- `engdocs-rag` (generic - reusable beyond Confluence later)

Recommendation: standalone new repo, portfolio-facing on its own merits - a general document-RAG capability that reads well independently, without needing another repo's context to make sense of it.

---

## 7. What We're Explicitly NOT Building Yet

- Real graph extraction/traversal (entities, relations, community summarization) - deferred as a future retrieval mode once hybrid+rerank+eval is solid and measured. Bolting on a real GraphRAG layer later, on top of a system that already has grounding checks and eval, will produce a much stronger result than building graph-first.
- Multi-tenant / multi-space support - single Confluence space (topic and space key are config, e.g. a personal "World LLMs" knowledge base you author yourself) for v1, matching current scope.
- Fine-tuned reranker - start with the small off-the-shelf cross-encoder, revisit only if eval numbers say retrieval precision is the bottleneck.
- Live/real-time indexing - nightly-refresh-via-Git is the free-tier trade-off; if that cadence becomes a real problem, that's the trigger to move to Chroma Cloud or a paid always-on host, not before.
- Qdrant / a dedicated vector DB - the earlier Qdrant design is still valid and worth revisiting if this ever needs to scale past what Chroma comfortably handles (a few thousand chunks, light query volume is where Chroma is genuinely fine). Nothing here is a dead end, just right-sized for now.

---

## 8. Build Order (as planned pre-code)

Kept as originally written, for historical context. Status per step, as of this doc's last update:

1. ✅ Confluence structure-aware parser + chunker (offline script, run locally first) - [`parsing.py`](parsing.py)
2. ✅ `config.py` reading `STORAGE_MODE` / `DATABASE_URL` / API keys from env - [`config.py`](config.py)
3. ✅ Reindex job: embed (OpenAI) → build Chroma `PersistentClient` index + `bm25_index.pkl`, respecting `STORAGE_MODE` - [`reindex.py`](reindex.py), [`vectorstore.py`](vectorstore.py)
4. ✅ SQLite schema for local dev (`page_versions`, `query_audit_log`, `eval_runs`) via SQLAlchemy, same schema portable to Postgres - [`db.py`](db.py)
5. ✅ Hybrid retrieval (Chroma dense + rank_bm25 sparse + RRF fusion) + small HF cross-encoder reranking - [`vectorstore.py`](vectorstore.py), [`reranker.py`](reranker.py)
6. ✅ Router (Claude, structured output) + generation + groundedness node (Claude) - [`llm.py`](llm.py), [`graph.py`](graph.py). Ordering changed from the original plan: retrieval runs before the router, not after (see §3.5, §5).
7. ✅ Ragas golden set + eval harness (run locally, results written to `eval_runs`) - [`eval/run_eval.py`](eval/run_eval.py), [`eval/golden_set.jsonl`](eval/golden_set.jsonl)
8. ✅ Streamlit UI - [`app.py`](app.py)
9. ⬜ Dockerfile - not yet added. Still the right call for "runs anywhere" via `docker run` on Render/Railway/Fly.io/a VPS; not required for the current local + Streamlit Community Cloud deployment targets.
10. ✅ Deployment target support: `STORAGE_MODE`/`DATABASE_URL` config knobs, plus a scheduled GitHub Action reindex trigger - [`.github/workflows/reindex.yml`](.github/workflows/reindex.yml)
11. ⬜ Observability wiring (LangFuse or OpenTelemetry) - not yet added; nothing beyond the `query_audit_log` table exists for tracing today. Old repos retired.
