import os
import time
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
# Must match the vector_size of the pgvector table (see embedder.setup_table).
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# Gemini free tier allows 100 embed requests/minute; stay under it and back off on 429s.
_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))
_RATE_LIMIT_SLEEP = 65
_MAX_RETRIES = 3


class GeminiEmbeddings(Embeddings):
    """Google Gemini embeddings pinned to a fixed dimensionality so the vectors
    fit the pgvector table, with retrieval-optimised task types and free-tier
    rate-limit handling."""

    def __init__(self):
        self._inner = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(_MAX_RETRIES):
            try:
                return self._inner.embed_documents(
                    batch,
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBED_DIM,
                )
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < _MAX_RETRIES - 1:
                    print(f"[Embeddings] Free-tier rate limit hit. Waiting {_RATE_LIMIT_SLEEP}s before retrying...")
                    time.sleep(_RATE_LIMIT_SLEEP)
                else:
                    raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            if i > 0:
                print(f"[Embeddings] Pausing {_RATE_LIMIT_SLEEP}s between batches to respect the free-tier quota "
                      f"({i}/{len(texts)} chunks done)...")
                time.sleep(_RATE_LIMIT_SLEEP)
            vectors.extend(self._embed_batch_with_retry(texts[i:i + _BATCH_SIZE]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(
            text,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBED_DIM,
        )


def get_embeddings() -> Embeddings:
    return GeminiEmbeddings()
