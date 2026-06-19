import os
import urllib.parse
from dotenv import load_dotenv
from langchain_postgres import PGEngine, PGVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
}
TABLE       = os.getenv("DB_TABLE")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

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


def retrieve(query: str, top_k: int = 3) -> list[str]:
    engine = get_engine()
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE,
        embedding_service=embeddings,
    )

    results = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]