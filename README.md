# RAG Demo — Agentic Retrieval-Augmented Generation

A LangGraph-powered RAG system with multi-tool routing, an LLM-as-judge evaluation loop, and a Human-in-the-Loop (HITL) fallback. Built with FastAPI, Ollama, PostgreSQL + pgvector, Tavily, and Groq.

---

## Architecture

```
User Query
    │
    ▼
┌─────────┐     ┌──────────────┐     ┌─────────────┐
│ Router  │────▶│ Vector Search│────▶│             │
│ (plans  │     └──────────────┘     │  Generator  │
│  tools) │────▶┌──────────────┐────▶│  (llama3.2) │
│         │     │  Web Search  │     │             │
│         │────▶└──────────────┘     └──────┬──────┘
│         │     ┌──────────────┐            │
│         │────▶│ GitHub Read  │            ▼
└─────────┘     └──────────────┘     ┌─────────────┐
                                     │    Judge    │
                                     │  (Groq 70B) │
                                     └──────┬──────┘
                              ACCEPT        │        RETRY (< 3x)
                         ┌─────────────────┤──────────────────┐
                         ▼                 │                  ▼
                        END           RETRY ≥ 3x         Rewrite Node
                                           │                  │
                                           ▼                  └──▶ Router
                                      HITL Node
                                    (logs to terminal)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| PostgreSQL | 14+ | With the `pgvector` extension enabled |
| Ollama | Latest | Running locally at `http://localhost:11434` |
| Groq API key | — | Free tier at [console.groq.com](https://console.groq.com) |
| Tavily API key | — | Free tier at [tavily.com](https://tavily.com) |
| GitHub Token | — | Optional — increases GitHub API rate limits |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd agentic-ai

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
cd app
pip install -r requirements.txt
```

### 3. Set up PostgreSQL with pgvector

Create a database and enable the extension:

```sql
CREATE DATABASE vectordb;
\c vectordb
CREATE EXTENSION IF NOT EXISTS vector;
```

### 4. Pull Ollama models

```bash
# Generator model (used for answering queries)
ollama pull llama3.2

# Embedding model (used for document ingestion and vector search)
ollama pull nomic-embed-text
```

### 5. Configure environment variables

Copy the example below into a `.env` file at the project root (`rag-demo/.env`):

```env
# PostgreSQL
DB_NAME=vectordb
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5433
DB_TABLE=documents

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
CHAT_MODEL=llama3.2

# Tavily (web search) — https://tavily.com
TAVILY_API_KEY=your_tavily_api_key_here

# GitHub (optional — raises API rate limits)
GITHUB_TOKEN=your_github_token_here

# Groq (judge model — free tier) — https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here
JUDGE_MODEL=llama-3.3-70b-versatile

# Langfuse (observability — optional, traces stay disabled if unset)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_BASE_URL=http://localhost:3000
```

---

## Running the API

```bash
cd app
uvicorn api:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`.

Startup log should show:
```
[Config] Judge model: Groq (llama-3.3-70b-versatile)
```

---

## API Endpoints

### `GET /health`
Check if the server is running.

```bash
curl http://127.0.0.1:8000/health
```

---

### `POST /ingest`
Upload a `.pdf` or `.docx` document to the vector store.

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "file=@documents/NimbusCart_Policy_Handbook.docx"
```

Response:
```json
{
  "filename": "NimbusCart_Policy_Handbook.docx",
  "characters_extracted": 42000,
  "chunks_stored": 84,
  "message": "Document ingested successfully."
}
```

---

### `POST /chat`
Ask a question. The system routes to the appropriate tools, generates an answer, evaluates it with the judge, and retries if needed.

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is NimbusCart'\''s return policy?"}'
```

Response:
```json
{
  "question": "What is NimbusCart's return policy?",
  "answer": "NimbusCart offers a 30-day return window...",
  "steps_taken": ["vector_search", "generate", "judge"],
  "context_sources": ["--- [Vector Search Result..."]
}
```

**`steps_taken` values explained:**

| Step | Description |
|---|---|
| `vector_search` | Searched the ingested document database |
| `web_search` | Searched the public web via Tavily |
| `github_read` | Fetched content from a GitHub repository |
| `generate` | Generated an answer using llama3.2 |
| `judge` | Evaluated the answer using Groq llama-3.3-70b |
| `rewrite` | Query was rewritten and retried |
| `hitl` | Max retries reached — flagged for human review |

---

## Running the Frontend

Install Streamlit (already in `requirements.txt`) and run it alongside the API:

```bash
# Terminal 1 — API
cd app
uvicorn api:app --reload --port 8000

# Terminal 2 — Frontend
cd app
streamlit run streamlit_app.py
```

The UI opens automatically at `http://localhost:8501`.

**Features:**
- Chat interface with full conversation history
- Document upload and ingestion (PDF / DOCX) from the sidebar
- Step badges on every response showing exactly which tools ran (`🔍 vector_search`, `🌐 web_search`, `⚖️ judge`, etc.)
- HITL warning shown inline when the judge could not get a satisfactory answer after max retries
- API health indicator in the sidebar
- Clear chat button

---

## Observability with Langfuse

Every `/chat` request is traced end-to-end (router → tools → generator → judge → rewrite/HITL) when Langfuse keys are set. Tracing is fully optional — if `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are unset, the app runs exactly as before with no Langfuse calls.

### Run Langfuse locally

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Open `http://localhost:3000`, create a project, and copy the generated public/secret keys into your `.env` (see above).

### What gets traced

- A full trace per `/chat` call, named `rag-chat`, tagged `rag-demo`
- Each LangGraph node (router, vector_search, web_search, github_read, generator, judge, rewrite, hitl) as a nested span, including LLM calls with prompts/completions and token usage
- A `judge_decision` score (1 = accept, 0 = retry) with the judge's reasoning attached as a comment — lets you filter traces in the Langfuse UI by where the judge struggled

---

## Visualising the Graph

Generate a Mermaid diagram of the LangGraph pipeline:

```bash
cd app
python generate_graph.py
```

This produces two outputs:

### Option 1 — mermaid.live (no install required)

1. Open [mermaid.live](https://mermaid.live) in your browser
2. The script prints the Mermaid code to the terminal and saves it to `graph_styled.txt`
3. Copy the contents of `graph_styled.txt` and paste it into the editor on the left
4. The diagram renders instantly on the right — you can export it as PNG or SVG from there

### Option 2 — VS Code extension

1. Install the **Markdown Preview Mermaid Support** extension in VS Code
2. Create a new file called `graph.md` and paste this block into it:
   ````
   ```mermaid
   <paste contents of graph_styled.txt here>
   ```
   ````
3. Open the Markdown preview (`Ctrl+Shift+V`) to see the diagram rendered inline

### Option 3 — PNG file (best effort)

`generate_graph.py` also tries to export `langgraph_flow.png` directly via the remote `mermaid.ink` API. This requires outbound internet access and will silently skip if blocked (e.g. by a corporate TLS-intercepting proxy) — in that case, just use Option 1 or 2 above instead.

---

## Project Structure

```
rag-demo/
├── .env                        # Environment variables (not committed)
├── documents/                  # Sample documents for ingestion
│   └── NimbusCart_Policy_Handbook.docx
└── app/
    ├── api.py                  # FastAPI routes (/health, /ingest, /chat)
    ├── generate_graph.py       # Graph visualisation script
    ├── requirements.txt        # Python dependencies
    └── core/
        ├── graph.py            # LangGraph nodes and workflow (main logic)
        ├── generator.py        # Legacy generator (used by api.py)
        ├── embedder.py         # Embedding and pgvector storage
        ├── retriever.py        # Vector similarity search
        ├── ingestor.py         # PDF / DOCX text extraction
        └── chunker.py          # Text chunking
```
