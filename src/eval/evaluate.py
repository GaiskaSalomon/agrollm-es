"""Evaluation suite for the RAG + fine-tuned model, with a hallucination check.

Test set (src/eval/testset.jsonl, 100 questions):
  * answerable / seen   : the gold document was used for fine-tuning.
  * answerable / unseen : the gold document was withheld from fine-tuning (it is still in the
                          RAG index), so the answer must come from retrieval, not memory.
  * unanswerable        : the corpus has no answer; the correct behaviour is to abstain.

Metrics per question:
  * keyword_recall  : fraction of `must_include` entries present in the answer. An entry can list
                      alternatives separated by "|"; whitespace and case are ignored.
  * retrieval_hit   : a retrieved chunk comes from a gold document (answerable questions only).
  * semantic_sim    : cosine similarity between answer and reference embeddings.
  * groundedness    : cosine similarity between answer and the retrieved context;
                      low groundedness flags a likely hallucination (answer not
                      supported by retrieved evidence).
  * hallucination   : groundedness < threshold.
  * abstained       : the answer says the context does not contain the answer.

Aggregates are reported per group, because pooling them would hide the failures.

Writes an aggregate JSON report. Runs in retrieval-only mode if no GPU/model, so the
RAG + grounding metrics still work without fine-tuning first.

Usage:
    python -m src.eval.evaluate --report outputs/eval_report.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.rag.embeddings import embed
from src.rag.query import answer

DEFAULT_TESTSET = Path(__file__).parent / "testset.jsonl"
DEFAULT_CORPUS = Path("data/raw/corpus.jsonl")
ABSTAIN_RE = re.compile(
    r"no (contiene|incluye|menciona|especifica|proporciona|hay informaci|se encuentra|se menciona|"
    r"puedo|dispongo|tengo)|el contexto no|no est[aá] (en|incluid)|sin informaci",
    re.IGNORECASE,
)


def _cos(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def keyword_recall(text: str, keys: list[str]) -> float:
    if not keys:
        return 1.0
    low = _norm(text)
    return sum(1 for k in keys if any(_norm(alt) in low for alt in k.split("|"))) / len(keys)


def abstains(text: str) -> bool:
    return bool(ABSTAIN_RE.search(text))


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "keyword_recall": _mean(rows, "keyword_recall"),
        "retrieval_hit": _mean(rows, "retrieval_hit"),
        "semantic_sim": _mean(rows, "semantic_sim"),
        "groundedness": _mean(rows, "groundedness"),
        "hallucination_rate": _mean(rows, "hallucination"),
        "abstention_rate": _mean(rows, "abstained"),
    }


def evaluate(testset: Path, base: str, adapter: str, corpus: Path = DEFAULT_CORPUS,
             halluc_threshold: float = 0.35) -> dict:
    cases = [json.loads(l) for l in testset.open(encoding="utf-8") if l.strip()]
    docs = {}
    for line in corpus.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            docs[rec["id"]] = rec["text"]
    results = []

    for case in cases:
        res = answer(case["question"], base, adapter)
        ans = res["answer"]
        ctx = " ".join(res["context"])

        ans_vec, ref_vec, ctx_vec = embed([ans, case["reference"], ctx or " "])
        groundedness = _cos(ans_vec, ctx_vec)
        answerable = case.get("kind", "answerable") == "answerable"
        gold = [docs[d] for d in case.get("gold_docs", []) if d in docs]
        hit = any(c.strip() in g for c in res["context"] for g in gold) if answerable else None
        results.append(
            {
                "question": case["question"],
                "kind": case.get("kind", "answerable"),
                "split": case.get("split", "n/a"),
                "mode": res["mode"],
                "answer": ans,
                "keyword_recall": round(keyword_recall(ans, case.get("must_include", [])), 3) if answerable else None,
                "retrieval_hit": hit,
                "semantic_sim": round(_cos(ans_vec, ref_vec), 3) if answerable else None,
                "groundedness": round(groundedness, 3),
                "hallucination": bool(groundedness < halluc_threshold),
                "abstained": abstains(ans),
            }
        )

    groups = {
        "answerable_seen": [r for r in results if r["kind"] == "answerable" and r["split"] == "seen"],
        "answerable_unseen": [r for r in results if r["kind"] == "answerable" and r["split"] == "unseen"],
        "unanswerable": [r for r in results if r["kind"] == "unanswerable"],
    }
    return {
        "n_cases": len(results),
        "mode": results[0]["mode"] if results else "n/a",
        "groups": {name: summarize(rows) for name, rows in groups.items()},
        "per_case": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RAG + model with hallucination check.")
    ap.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="outputs/qwen-agro-lora")
    ap.add_argument("--report", type=Path, default=Path("outputs/eval_report.json"))
    ap.add_argument("--halluc-threshold", type=float, default=0.35)
    args = ap.parse_args()

    report = evaluate(args.testset, args.base, args.adapter, args.corpus, args.halluc_threshold)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[eval] mode={report['mode']}  cases={report['n_cases']}")
    for name, g in report["groups"].items():
        print(f"  {name} (n={g['n']}): " + ", ".join(f"{k}={v}" for k, v in g.items() if k != "n" and v is not None))
    print(f"[eval] report -> {args.report}")


if __name__ == "__main__":
    main()
