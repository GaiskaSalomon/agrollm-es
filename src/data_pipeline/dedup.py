"""Deduplicate a corpus: exact (hash) + near-duplicate (MinHash + LSH).

Near-duplicate detection uses character n-gram shingles fed to MinHash, then an
LSH index to find pairs above a Jaccard threshold. This is the standard approach
for large web corpora where verbatim dedup is not enough.

Falls back to exact-only dedup if `datasketch` is not installed.

Usage:
    python -m src.data_pipeline.dedup --in data/interim/clean.jsonl \
                                      --out data/interim/dedup.jsonl --threshold 0.8
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from datasketch import MinHash, MinHashLSH

    _HAS_DATASKETCH = True
except ImportError:  # graceful degradation
    _HAS_DATASKETCH = False


def _shingles(text: str, k: int = 5) -> set[str]:
    text = text.lower()
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _minhash(text: str, num_perm: int = 128) -> "MinHash":
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def run(in_path: Path, out_path: Path, threshold: float = 0.8) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    n_in = len(records)

    # 1) Exact dedup by normalized-text hash.
    seen_hashes: set[str] = set()
    exact_unique = []
    for rec in records:
        h = hashlib.sha1(rec["text"].strip().lower().encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            exact_unique.append(rec)

    # 2) Near-duplicate dedup via MinHash + LSH.
    if _HAS_DATASKETCH:
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        kept = []
        for i, rec in enumerate(exact_unique):
            m = _minhash(rec["text"])
            if lsh.query(m):  # a near-duplicate already indexed
                continue
            lsh.insert(str(i), m)
            kept.append(rec)
    else:
        print("[dedup] datasketch not installed -> exact dedup only")
        kept = exact_unique

    with out_path.open("w", encoding="utf-8") as fout:
        for rec in kept:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return n_in, len(kept)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deduplicate a JSONL corpus.")
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="Jaccard threshold for near-duplicates (0-1).")
    args = ap.parse_args()

    n_in, n_out = run(args.in_path, args.out_path, args.threshold)
    print(f"[dedup] {n_in} docs -> {n_out} unique ({n_in - n_out} removed)")


if __name__ == "__main__":
    main()
