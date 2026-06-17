"""Ingest a corpus into pgvector: chunk -> embed -> store.

Usage:
    docker compose up -d
    python -m src.rag.ingest --in data/interim/dedup.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rag import db
from src.rag.embeddings import embed


def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> list[str]:
    """Split into overlapping character windows on sentence-ish boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # try to break at the last period within the window
        dot = text.rfind(". ", start, end)
        if dot != -1 and dot > start + max_chars // 2:
            end = dot + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def run(in_path: Path) -> int:
    rows = []
    for line in in_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for ch in chunk_text(rec["text"]):
            rows.append((rec.get("id", "?"), rec.get("source", ""), ch))

    # Load the embedding model (initializes torch's native libs) BEFORE opening the
    # psycopg connection. On Windows, importing torch after libpq can segfault due to
    # a native DLL (OpenMP) conflict; doing torch first avoids it.
    contents = [r[2] for r in rows]
    vecs = embed(contents)

    conn = db.get_conn()
    db.reset(conn)
    db.init_schema(conn)

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (doc_id, source, content, embedding) VALUES (%s, %s, %s, %s)",
            [(r[0], r[1], r[2], vecs[i]) for i, r in enumerate(rows)],
        )
    print(f"[ingest] stored {len(rows)} chunks into pgvector")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest corpus into pgvector.")
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    args = ap.parse_args()
    run(args.in_path)


if __name__ == "__main__":
    main()
