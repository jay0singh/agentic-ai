import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from core.ingestor import load_document
from core.chunker import chunk_text
from core.embedder import setup_table, embed_and_store
from core.retriever import retrieve
from core.generator import run_orchestrator

app = FastAPI(
    title="RAG Chatbot API",
    description="Upload documents and query them using local LLMs via Ollama.",
    version="1.0.0"
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    top_k: int = 3


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
async def ingest(file: UploadFile = File(...)):
    """
    Upload a .pdf or .docx file.
    Extracts text, chunks it, embeds it, and stores in pgvector.
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
        embed_and_store(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "filename": filename,
        "characters_extracted": len(text),
        "chunks_stored": len(chunks),
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
        state = run_orchestrator(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "question": request.question,
        "answer": state.get("response", "No response generated."),
        "steps_taken": state.get("steps_taken", []),
        "context_sources": state.get("context", []),
        "judge_log": state.get("judge_log", [])
    }