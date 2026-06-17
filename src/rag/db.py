"""PostgreSQL + pgvector helpers: connection, schema, vector search.

Connection settings come from env vars (see docker-compose.yml defaults):
    PGHOST=localhost PGPORT=5432 PGUSER=agro PGPASSWORD=agro PGDATABASE=agrollm
"""
from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector

EMBED_DIM = 384  # matches sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        user=os.getenv("PGUSER", "agro"),
        password=os.getenv("PGPASSWORD", "agro"),
        dbname=os.getenv("PGDATABASE", "agrollm"),
        autocommit=True,
    )
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection, dim: int = EMBED_DIM) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id        BIGSERIAL PRIMARY KEY,
            doc_id    TEXT NOT NULL,
            source    TEXT,
            content   TEXT NOT NULL,
            embedding VECTOR({dim})
        )
        """
    )
    # HNSW index for approximate nearest-neighbour search (cosine distance).
    # Preferred over IVFFlat here: it gives high recall regardless of corpus size,
    # whereas IVFFlat needs many rows per list or it returns empty probes on small data.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )


def reset(conn: psycopg.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS chunks")


def search(conn: psycopg.Connection, query_vec, k: int = 4):
    """Return the k most similar chunks as (content, source, distance)."""
    cur = conn.execute(
        """
        SELECT content, source, embedding <=> %s AS distance
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (query_vec, query_vec, k),
    )
    return cur.fetchall()
