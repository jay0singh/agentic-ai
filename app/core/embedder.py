import os
import urllib.parse
import psycopg
from dotenv import load_dotenv
from langchain_postgres import PGEngine, PGVectorStore
from langchain_core.documents import Document

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


def get_vector_store():
    engine = get_engine()
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE,
        embedding_service=get_embeddings(),
    )


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


def embed_and_store(chunks: list[str], source: str = "unknown") -> None:
    print(f"Embedding {len(chunks)} chunks using '{EMBED_MODEL}' via Gemini API...")
    vector_store = get_vector_store()

    # Convert chunks to LangChain Document objects, tagging each with its
    # source file so re-ingesting the same file can replace old chunks.
    documents = [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]
    vector_store.add_documents(documents)
    print("All chunks embedded and stored.")