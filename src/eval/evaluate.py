"""Evaluation suite for the RAG + fine-tuned model, with a hallucination check.

Metrics per question:
  * keyword_recall  : fraction of `must_include` tokens present in the answer.
  * semantic_sim    : cosine similarity between answer and reference embeddings.
  * groundedness    : cosine similarity between answer and the retrieved context;
                      low groundedness flags a likely hallucination (answer not
                      supported by retrieved evidence).
  * hallucination   : groundedness < threshold.

Writes an aggregate JSON report. Runs in retrieval-only mode if no GPU/model, so the
RAG + grounding metrics still work without fine-tuning first.

Usage:
    python -m src.eval.evaluate --report outputs/eval_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.rag.embeddings import embed
from src.rag.query import answer

DEFAULT_TESTSET = Path(__file__).parent / "testset.jsonl"


def _cos(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def keyword_recall(text: str, keys: list[str]) -> float:
    if not keys:
        return 1.0
    low = text.lower()
    return sum(1 for k in keys if k.lower() in low) / len(keys)


def evaluate(testset: Path, base: str, adapter: str, halluc_threshold: float = 0.35) -> dict:
    cases = [json.loads(l) for l in testset.open(encoding="utf-8") if l.strip()]
    results = []

    for case in cases:
        res = answer(case["question"], base, adapter)
        ans = res["answer"]
        ctx = " ".join(res["context"])

        ans_vec, ref_vec, ctx_vec = embed([ans, case["reference"], ctx or " "])
        groundedness = _cos(ans_vec, ctx_vec)
        results.append(
            {
                "question": case["question"],
                "mode": res["mode"],
                "keyword_recall": round(keyword_recall(ans, case.get("must_include", [])), 3),
                "semantic_sim": round(_cos(ans_vec, ref_vec), 3),
                "groundedness": round(groundedness, 3),
                "hallucination": bool(groundedness < halluc_threshold),
            }
        )

    n = len(results)
    agg = {
        "n_cases": n,
        "avg_keyword_recall": round(sum(r["keyword_recall"] for r in results) / n, 3),
        "avg_semantic_sim": round(sum(r["semantic_sim"] for r in results) / n, 3),
        "avg_groundedness": round(sum(r["groundedness"] for r in results) / n, 3),
        "hallucination_rate": round(sum(r["hallucination"] for r in results) / n, 3),
        "mode": results[0]["mode"] if results else "n/a",
        "per_case": results,
    }
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RAG + model with hallucination check.")
    ap.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="outputs/qwen-agro-lora")
    ap.add_argument("--report", type=Path, default=Path("outputs/eval_report.json"))
    ap.add_argument("--halluc-threshold", type=float, default=0.35)
    args = ap.parse_args()

    report = evaluate(args.testset, args.base, args.adapter, args.halluc_threshold)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[eval] mode={report['mode']}  cases={report['n_cases']}")
    print(f"  keyword_recall   : {report['avg_keyword_recall']}")
    print(f"  semantic_sim     : {report['avg_semantic_sim']}")
    print(f"  groundedness     : {report['avg_groundedness']}")
    print(f"  hallucination    : {report['hallucination_rate']}")
    print(f"[eval] report -> {args.report}")


if __name__ == "__main__":
    main()
