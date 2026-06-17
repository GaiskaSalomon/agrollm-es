"""Clean and normalize a Spanish text corpus.

Steps:
  * Unicode NFC normalization and whitespace collapse.
  * Strip control characters and fix common mojibake.
  * Quality filters: min length, max symbol ratio, language heuristic (Spanish).

Input  : JSONL with at least {"id", "text"}.
Output : JSONL with the same fields plus cleaned "text".

Usage:
    python -m src.data_pipeline.clean --in data/raw/corpus.jsonl \
                                      --out data/interim/clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

# A few high-frequency Spanish stopwords used as a cheap language heuristic.
_ES_MARKERS = {
    "de", "la", "el", "que", "en", "y", "los", "las", "del", "se", "por",
    "con", "para", "una", "un", "es", "su", "al", "como", "más", "o",
}

_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MOJIBAKE = {"Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú", "Ã±": "ñ"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for bad, good in _MOJIBAKE.items():
        text = text.replace(bad, good)
    text = _CTRL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def is_spanish(text: str, min_hits: int = 3) -> bool:
    tokens = {t for t in re.findall(r"[a-záéíóúñ]+", text.lower())}
    return len(tokens & _ES_MARKERS) >= min_hits


def symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    non_alnum = sum(1 for c in text if not (c.isalnum() or c.isspace()))
    return non_alnum / len(text)


def keep(text: str, min_chars: int = 80, max_symbol_ratio: float = 0.25) -> bool:
    return (
        len(text) >= min_chars
        and symbol_ratio(text) <= max_symbol_ratio
        and is_spanish(text)
    )


def run(in_path: Path, out_path: Path) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            rec["text"] = normalize(rec.get("text", ""))
            if keep(rec["text"]):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
    return n_in, n_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean a Spanish JSONL corpus.")
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    args = ap.parse_args()

    n_in, n_out = run(args.in_path, args.out_path)
    print(f"[clean] read {n_in} docs -> kept {n_out} ({n_in - n_out} filtered)")


if __name__ == "__main__":
    main()
