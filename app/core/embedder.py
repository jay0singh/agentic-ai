import os
import urllib.parse
from dotenv import load_dotenv
from langchain_postgres import PGEngine, PGVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

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


def get_vector_store():
    engine = get_engine()
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE,
        embedding_service=embeddings,
    )


def setup_table():
    engine = get_engine()
    try:
        # Initialize the pgvector table. nomic-embed-text generates 768-dimensional embeddings.
        engine.init_vectorstore_table(
            table_name=TABLE,
            vector_size=768,
        )
        print(f"Table '{TABLE}' is ready.")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Table '{TABLE}' already exists. Skipping initialization.")
        else:
            raise e


def embed_and_store(chunks: list[str]) -> None:
    print(f"Embedding {len(chunks)} chunks using '{EMBED_MODEL}' via LangChain...")
    vector_store = get_vector_store()

    # Convert chunks to LangChain Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store.add_documents(documents)
    print("All chunks embedded and stored.")