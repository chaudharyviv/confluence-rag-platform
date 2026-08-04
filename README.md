# Confluence RAG Platform

A production‑shaped Retrieval‑Augmented Generation (RAG) platform for your Confluence documentation. It combines dense vector similarity (Chroma) and sparse BM25 search, re‑ranks results with a cross‑encoder, and routes generation via Claude for high‑quality answers.

A production-shaped RAG system over a Confluence space you author yourself.
Hybrid retrieval (Chroma dense + BM25 sparse, RRF-fused) → small HuggingFace
cross-encoder reranking → Claude-based routing and generation → a
groundedness check before anything is shown as "answered from the knowledge
base." See `production-rag-architecture.md` for the full design rationale -
this README is just setup.

## Why this exists

Two earlier repos (`Confluence-Rag`, `Confluence-Serpapi-GraphRag`) had real
bugs: a vector store that didn't actually persist, keyword-based domain
routing, and a "graph" that was metadata no one ever traversed. This is the
production rebuild - see the architecture doc for the specifics.

## Installation

```bash
git clone <this-repo>
cd confluence-rag-platform
pip install -r requirements.txt
cp .env.example .env
# fill in .env: OPENAI_API_KEY, ANTHROPIC_API_KEY, CONFLUENCE_* vars
```

Build the index (run this before first launching the app):

```bash
python reindex.py
```

## Usage

After installing, you can run the application locally with:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to interact with the RAG UI. The same command works on a VPS, Docker container, or Streamlit Community Cloud.

## Storage modes

Everything host-specific is one env var, `STORAGE_MODE`, never a code fork:

- **`local`** (default) - `reindex.py` writes directly to `./chroma_db/`.
  Use this anywhere with a real persistent disk: your laptop, a VPS, a Docker
  volume.
- **`git`** - `reindex.py` commits `chroma_db/` + `bm25_index.pkl` back to the
  repo after building them. Use this on ephemeral-disk hosts (Streamlit
  Community Cloud is the main one) - auto-redeploy on push picks up the
  fresh index. See `.github/workflows/reindex.yml` for a scheduled version.

## Audit / eval database

Defaults to local SQLite (`DATABASE_URL=sqlite:///./local.db`) - zero setup,
appropriate for a personal project where losing history on a sleep/restart
is a fine trade-off. Swap `DATABASE_URL` to a Neon Postgres connection string
only if you want audit history to survive across deployed sessions. Don't use
Supabase for this - its free tier pauses the whole project after 7 days
idle and needs a manual unpause; Neon auto-resumes on the next query with no
manual step.

## Evaluation

```bash
pip install -r eval/requirements-eval.txt
python eval/run_eval.py eval/golden_set.jsonl
```

Write your own golden set as you author Confluence pages - a couple of QA
pairs per page, with real ground truth, is a much better eval-authoring
workflow than reverse-engineering questions from someone else's docs.
`eval/golden_set.jsonl` includes two edge-case rows (an out-of-date question,
an out-of-domain question) meant to be eyeballed against the printed
`router=` output rather than scored as faithfulness/precision numbers -
Ragas' metrics assume an in-domain, answerable question.

## What's deliberately not built yet

Real graph extraction/traversal, multi-space support, a fine-tuned reranker,
live (non-batch) indexing. See §7 of the architecture doc for why, and what
would trigger building them.

## Config reference

See `.env.example` for every setting, with inline comments.

## Contributing

We welcome contributions! Please fork the repository and submit a pull request. Follow these steps:

1. **Set up** – run the installation steps above.
2. **Create a branch** – `git checkout -b my-feature`.
3. **Make changes** – ensure code follows existing style and passes tests.
4. **Run tests** – `pytest` (if tests are added).
5. **Submit PR** – describe the changes and reference any related issues.

For major changes, open an issue first to discuss the proposed design.
