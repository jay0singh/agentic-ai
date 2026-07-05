import json
import os
import urllib.parse
import uuid
from datetime import datetime, timezone

import psycopg
from dotenv import load_dotenv
from langchain_postgres import PGEngine

from core.embeddings import get_embeddings, EMBED_MODEL, EMBED_DIM

load_dotenv()

DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
}
TABLE       = os.getenv("DB_TABLE")

# Construct database connection URL (uses psycopg3 driver via 'psycopg')
db_user = DB_CONFIG["user"]
db_pass = urllib.parse.quote_plus(DB_CONFIG["password"] or "")
db_host = DB_CONFIG["host"]
db_port = DB_CONFIG["port"]
db_name = DB_CONFIG["dbname"]
CONNECTION_URL = f"postgresql+psycopg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = PGEngine.from_connection_string(url=CONNECTION_URL)
    return _engine


def setup_table():
    engine = get_engine()
    try:
        # Initialize the pgvector table. Vector size must match the Gemini
        # embedding dimensionality configured in core.embeddings.
        engine.init_vectorstore_table(
            table_name=TABLE,
            vector_size=EMBED_DIM,
        )
        print(f"Table '{TABLE}' is ready.")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Table '{TABLE}' already exists. Skipping initialization.")
        else:
            raise e

    # Full-text index used by the hybrid (keyword) half of retrieval.
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS "{TABLE}_content_fts" ON "{TABLE}" '
                f"USING GIN (to_tsvector('english', content))"
            )


def delete_by_source(source: str) -> int:
    """Remove all chunks previously ingested from the given source file.

    Returns the number of deleted rows. Lets a document be re-ingested
    without duplicating its chunks in the vector store."""
    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'DELETE FROM "{TABLE}" WHERE langchain_metadata->>%s = %s',
                    ("source", source),
                )
                return cur.rowcount
    except psycopg.errors.UndefinedTable:
        return 0


def list_documents() -> list[dict]:
    """Summarise ingested documents: source filename, chunk count, ingest time."""
    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT langchain_metadata->>%s, COUNT(*), MAX(langchain_metadata->>%s) '
                    f'FROM "{TABLE}" GROUP BY 1 ORDER BY 1',
                    ("source", "ingested_at"),
                )
                return [
                    {"source": source or "unknown", "chunks": chunks, "ingested_at": ingested_at}
                    for source, chunks, ingested_at in cur.fetchall()
                ]
    except psycopg.errors.UndefinedTable:
        return []


def embed_and_store(chunks: list[str], source: str = "unknown") -> int:
    """Embed chunks, then atomically replace the source's existing rows.

    Embedding (slow, can fail on rate limits) happens BEFORE anything is
    deleted, and the delete+insert runs in one transaction — so the old
    version of a document stays searchable until the new one fully lands,
    and a failed embed loses nothing. Returns the number of replaced rows."""
    print(f"Embedding {len(chunks)} chunks using '{EMBED_MODEL}' via Gemini API...")
    vectors = get_embeddings().embed_documents(chunks)

    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = json.dumps({"source": source, "ingested_at": ingested_at})

    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f'DELETE FROM "{TABLE}" WHERE langchain_metadata->>%s = %s',
                ("source", source),
            )
            replaced = cur.rowcount
            for chunk, vector in zip(chunks, vectors):
                cur.execute(
                    f'INSERT INTO "{TABLE}" (langchain_id, content, embedding, langchain_metadata) '
                    f"VALUES (%s, %s, %s::vector, %s)",
                    (str(uuid.uuid4()), chunk, str(vector), metadata),
                )

    print(f"All chunks embedded and stored ({replaced} old chunks replaced).")
    return replaced