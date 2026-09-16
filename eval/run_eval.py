"""
Ragas eval harness. Run locally or in CI:

    python eval/run_eval.py eval/golden_set.jsonl

Each line in the golden set is a hand-written QA pair - the whole point of
authoring the knowledge base yourself is that you can write these as you go,
with actual ground truth, instead of reverse-engineering questions from
someone else's docs.

Requires eval/requirements-eval.txt (kept separate from the main app's
requirements so the deployed Streamlit app doesn't carry ragas/langchain's
weight - eval only needs to run locally or in CI, never in production).

Not every row is scored the same way:
  - out_of_domain / needs_external rows have no "ground truth answer" for
    Ragas to compare against (per Ragas' own assumption that the question is
    answerable) - these are asserted against router_decision instead, and
    never enter the Ragas call.
  - exact_lookup rows carry a structured `expected_fields` dict, checked with
    plain normalized comparison instead of (or alongside) the LLM-judged
    Ragas metrics - cheaper, deterministic, reproducible.
  - Any row with `required_sources` gets a deterministic context-recall
    check: did retrieval actually surface those chunk ids, independent of
    Ragas' own (LLM-judged) context_recall metric.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db
from graph import ask

# Categories Ragas can't meaningfully score: there's no answerable ground
# truth to compare against, only a routing decision to check.
NOT_RAGAS_SCORABLE = {"out_of_domain", "needs_external"}


def load_golden_set(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize(text: str) -> str:
    return re.sub(r"[,\s]", "", text.strip().lower())


def check_exact_lookup(answer: str, expected_fields: dict) -> bool:
    """Deterministic pass/fail for exact_lookup rows: normalized substring
    match against the answer text, independent of any LLM judge."""
    normalized_answer = _normalize(answer)
    value = str(expected_fields.get("value", "")).strip()
    if not value or _normalize(value) not in normalized_answer:
        return False
    unit = expected_fields.get("unit")
    if unit and _normalize(str(unit)) not in normalized_answer:
        return False
    return True


def check_source_coverage(retrieved_ids: list[str], required_sources: list[str]) -> bool:
    """Deterministic pass/fail: did retrieval actually surface every chunk id
    the golden row says the answer must draw from."""
    return set(required_sources).issubset(set(retrieved_ids))


def run(golden_set_path: str, git_sha: str | None = None) -> None:
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from datasets import Dataset
    except ImportError:
        print(
            "ragas isn't installed. Run: pip install --break-system-packages -r "
            "eval/requirements-eval.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    db.init_db()
    golden = load_golden_set(golden_set_path)
    print(f"Loaded {len(golden)} golden QA pairs from {golden_set_path}")

    rows = []
    for item in golden:
        result = ask(item["question"])
        retrieved_ids = [c.chunk_id for c in result.get("reranked", [])]
        rows.append(
            {
                "question": item["question"],
                "ground_truth": item["expected_answer"],
                "answer": result["answer"],
                "contexts": [c.content for c in result.get("reranked", [])] or [""],
                "retrieved_ids": retrieved_ids,
                "router_decision": result["router_decision"],
                "category": item.get("category", "uncategorized"),
                "required_sources": item.get("required_sources", []),
                "answerable": item.get("answerable"),
                "expected_fields": item.get("expected_fields"),
            }
        )
        print(f"  ran: {item['question'][:60]!r} -> router={result['router_decision']}")

    # ---------- deterministic checks (no LLM judge involved) ----------

    exact_lookup_results = []
    for row in rows:
        if row["category"] == "exact_lookup":
            passed = check_exact_lookup(row["answer"], row["expected_fields"] or {})
            row["exact_lookup_pass"] = passed
            exact_lookup_results.append((row["question"], passed))

    router_match_results = []
    for row in rows:
        if row["category"] in NOT_RAGAS_SCORABLE:
            passed = row["router_decision"] == row["category"]
            row["router_match_pass"] = passed
            router_match_results.append((row["question"], passed))

    source_coverage_results = []
    for row in rows:
        if row["required_sources"]:
            passed = check_source_coverage(row["retrieved_ids"], row["required_sources"])
            row["source_coverage_pass"] = passed
            source_coverage_results.append((row["question"], passed))

    # ---------- Ragas (LLM-judged) metrics, only for scorable rows ----------

    ragas_rows = [r for r in rows if r["category"] not in NOT_RAGAS_SCORABLE]
    metric_cols = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    scores: dict = {}

    if ragas_rows:
        dataset = Dataset.from_dict(
            {
                "question": [r["question"] for r in ragas_rows],
                "answer": [r["answer"] for r in ragas_rows],
                "contexts": [r["contexts"] for r in ragas_rows],
                "ground_truth": [r["ground_truth"] for r in ragas_rows],
            }
        )
        result = evaluate(
            dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        )
        df = result.to_pandas()
        for i, row in enumerate(ragas_rows):
            for col in metric_cols:
                if col in df:
                    row[col] = df.iloc[i][col]

        metric_cols = [c for c in metric_cols if c in df]
        scores = df[metric_cols].mean(numeric_only=True).to_dict()
        print("\nRagas results (aggregate, scorable rows only):")
        for k, v in scores.items():
            print(f"  {k}: {v:.3f}")

        print("\nRagas results by category:")
        df["category"] = [r["category"] for r in ragas_rows]
        for category, group in df.groupby("category"):
            print(f"  {category} (n={len(group)}):")
            means = group[metric_cols].mean(numeric_only=True)
            for k, v in means.items():
                print(f"    {k}: {v:.3f}")
    else:
        print("\nNo rows scorable by Ragas (all rows are out_of_domain/needs_external).")

    # ---------- deterministic check summaries ----------

    def _print_summary(label: str, results: list[tuple[str, bool]]) -> None:
        if not results:
            return
        passed = sum(1 for _, ok in results if ok)
        print(f"\n{label}: {passed}/{len(results)} passed")
        for question, ok in results:
            if not ok:
                print(f"  FAIL: {question}")

    _print_summary("Exact-lookup checks", exact_lookup_results)
    _print_summary("Router-decision checks", router_match_results)
    _print_summary("Source-coverage checks", source_coverage_results)

    db.record_eval_run(
        num_questions=len(golden),
        context_precision=scores.get("context_precision", 0.0),
        context_recall=scores.get("context_recall", 0.0),
        faithfulness=scores.get("faithfulness", 0.0),
        answer_relevancy=scores.get("answer_relevancy", 0.0),
        raw_results=rows,
        git_sha=git_sha,
    )
    print("\nSaved to eval_runs table.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("golden_set", help="Path to a .jsonl golden QA set")
    parser.add_argument("--git-sha", default=None)
    args = parser.parse_args()
    run(args.golden_set, git_sha=args.git_sha)
