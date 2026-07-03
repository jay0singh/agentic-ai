import os
import urllib.parse
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


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Return up to top_k chunks as dicts with content, source filename and
    cosine distance, dropping weak matches above RETRIEVAL_MAX_DISTANCE."""
    engine = get_engine()
    vector_store = PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE,
        embedding_service=get_embeddings(),
    )

    results = vector_store.similarity_search_with_score(query, k=top_k)
    chunks = []
    for doc, distance in results:
        if distance > RETRIEVAL_MAX_DISTANCE:
            print(f"  [Retriever] Dropped weak match (distance {distance:.3f} > {RETRIEVAL_MAX_DISTANCE:.2f}).")
            continue
        metadata = doc.metadata or {}
        chunks.append({
            "content": doc.page_content,
            "source": metadata.get("source", "unknown"),
            "distance": float(distance),
        })
    return chunks