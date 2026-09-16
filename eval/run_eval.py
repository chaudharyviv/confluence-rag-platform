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
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db
from graph import ask


def load_golden_set(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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

    questions, ground_truths, answers, contexts = [], [], [], []
    categories, required_sources_list, answerable_list = [], [], []

    for item in golden:
        result = ask(item["question"])
        questions.append(item["question"])
        ground_truths.append(item["expected_answer"])
        answers.append(result["answer"])
        contexts.append([c.content for c in result.get("reranked", [])] or [""])
        categories.append(item.get("category", "uncategorized"))
        required_sources_list.append(item.get("required_sources", []))
        answerable_list.append(item.get("answerable"))
        print(f"  ran: {item['question'][:60]!r} -> router={result['router_decision']}")

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )
    df = result.to_pandas()
    df["category"] = categories
    df["required_sources"] = required_sources_list
    df["answerable"] = answerable_list

    metric_cols = [c for c in ("context_precision", "context_recall", "faithfulness", "answer_relevancy") if c in df]
    scores = df[metric_cols].mean(numeric_only=True).to_dict()
    print("\nEval results (aggregate):")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    print("\nEval results by category:")
    for category, group in df.groupby("category"):
        print(f"  {category} (n={len(group)}):")
        means = group[metric_cols].mean(numeric_only=True)
        for k, v in means.items():
            print(f"    {k}: {v:.3f}")

    db.record_eval_run(
        num_questions=len(golden),
        context_precision=scores.get("context_precision", 0.0),
        context_recall=scores.get("context_recall", 0.0),
        faithfulness=scores.get("faithfulness", 0.0),
        answer_relevancy=scores.get("answer_relevancy", 0.0),
        raw_results=df.to_dict(orient="records"),
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
