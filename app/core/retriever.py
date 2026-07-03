import os
import urllib.parse

import psycopg
from dotenv import load_dotenv
from langchain_postgres import PGEngine, PGVectorStore

from core.embeddings import get_embeddings

load_dotenv()

DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
}
TABLE       = os.getenv("DB_TABLE")

# Cosine distance cutoff (0 = identical, 2 = opposite). Matches scoring worse
# than this are dropped so irrelevant chunks don't pollute the generator prompt.
# Calibrated on gemini-embedding-001: on-topic queries score ~0.20-0.30,
# off-topic ones ~0.48+.
RETRIEVAL_MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.42"))

# Reciprocal Rank Fusion constant — standard value from the original RRF paper.
RRF_K = 60

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


def _keyword_search(query: str, top_k: int) -> list[dict]:
    """Postgres full-text search over chunk content, ranked by ts_rank.
    Catches exact-keyword queries ("section 7.4", product codes) that
    embeddings can be fuzzy about."""
    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT content, langchain_metadata->>%s FROM "{TABLE}" '
                    f"WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', %s) "
                    f"ORDER BY ts_rank(to_tsvector('english', content), websearch_to_tsquery('english', %s)) DESC "
                    f"LIMIT %s",
                    ("source", query, query, top_k),
                )
                return [
                    {"content": content, "source": source or "unknown", "distance": None}
                    for content, source in cur.fetchall()
                ]
    except psycopg.errors.UndefinedTable:
        return []
    except Exception as e:
        print(f"  [Retriever] Keyword search failed ({e}). Using vector results only.")
        return []


def _rrf_merge(vector_hits: list[dict], keyword_hits: list[dict], top_k: int) -> list[dict]:
    """Merge two ranked hit lists with Reciprocal Rank Fusion:
    score(chunk) = sum over lists of 1 / (RRF_K + rank). Chunks found by both
    searches rank highest; keyword-only hits keep distance=None."""
    scores: dict[str, float] = {}
    info: dict[str, dict] = {}
    for hits in (vector_hits, keyword_hits):
        for rank, hit in enumerate(hits):
            key = hit["content"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            entry = info.setdefault(key, dict(hit))
            if entry.get("distance") is None and hit.get("distance") is not None:
                entry["distance"] = hit["distance"]
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [info[key] for key in ordered]


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Hybrid retrieval: vector similarity + full-text keyword search, fused
    with RRF. Returns up to top_k dicts with content, source and cosine
    distance (None for keyword-only matches)."""
    engine = get_engine()
    vector_store = PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE,
        embedding_service=get_embeddings(),
    )

    vector_hits = []
    for doc, distance in vector_store.similarity_search_with_score(query, k=top_k):
        if distance > RETRIEVAL_MAX_DISTANCE:
            print(f"  [Retriever] Dropped weak match (distance {distance:.3f} > {RETRIEVAL_MAX_DISTANCE:.2f}).")
            continue
        metadata = doc.metadata or {}
        vector_hits.append({
            "content": doc.page_content,
            "source": metadata.get("source", "unknown"),
            "distance": float(distance),
        })

    keyword_hits = _keyword_search(query, top_k)
    return _rrf_merge(vector_hits, keyword_hits, top_k)