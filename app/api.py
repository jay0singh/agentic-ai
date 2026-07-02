import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import shutil
import tempfile
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from core.ingestor import load_document
from core.chunker import chunk_text
from core.embedder import setup_table, embed_and_store, delete_by_source
from core.retriever import retrieve
from core.generator import run_orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Flush any queued Langfuse events before the process exits so traces aren't lost.
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import get_client
            get_client().shutdown()
        except Exception as e:
            print(f"[Config] Langfuse shutdown failed: {e}")


app = FastAPI(
    title="RAG Chatbot API",
    description="Upload documents and query them using free cloud LLMs (Groq + Gemini embeddings).",
    version="1.0.0",
    lifespan=lifespan
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    top_k: int = 3
    session_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    steps_taken: list[str]
    context_sources: list[str]
    judge_log: list[str]

class HealthResponse(BaseModel):
    status: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Check if the API is running."""
    return {"status": "ok"}


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    """
    Upload a .pdf or .docx file.
    Extracts text, chunks it, embeds it, and stores in pgvector.
    Re-ingesting a file replaces its previously stored chunks.

    Sync on purpose: FastAPI runs it in a threadpool, so slow embedding
    (including free-tier rate-limit pauses) doesn't block the event loop.
    """
    # Validate file type
    filename = file.filename or ""
    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        raise HTTPException(
            status_code=400,
            detail="Only .pdf and .docx files are supported."
        )

    # Save uploaded file to a temp location so we can read it
    suffix = ".pdf" if filename.endswith(".pdf") else ".docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Run the ingestion pipeline
        text = load_document(tmp_path)
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        setup_table()
        replaced = delete_by_source(filename)
        if replaced:
            print(f"[Ingest] Replacing {replaced} existing chunks for '{filename}'.")
        embed_and_store(chunks, source=filename)
    except Exception:
        # Log the full error server-side only — raw exception text can leak
        # connection strings or other internals to the client.
        print(f"[Ingest] Failed for '{filename}':\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Ingestion failed. Check the server logs for details."
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return {
        "filename": filename,
        "characters_extracted": len(text),
        "chunks_stored": len(chunks),
        "chunks_replaced": replaced,
        "message": "Document ingested successfully."
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question. Returns the answer, steps taken, and accumulated context.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        state = run_orchestrator(
            request.question,
            session_id=request.session_id,
            user_id=request.user_id,
        )
    except Exception:
        print(f"[Chat] Failed to answer question:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer. Check the server logs for details."
        )

    return {
        "question": request.question,
        "answer": state.get("response", "No response generated."),
        "steps_taken": state.get("steps_taken", []),
        "context_sources": state.get("context", []),
        "judge_log": state.get("judge_log", [])
    }