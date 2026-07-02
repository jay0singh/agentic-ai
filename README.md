# RAG Demo — Agentic Retrieval-Augmented Generation

A LangGraph-powered RAG system with multi-tool routing, an LLM-as-judge evaluation loop, and a Human-in-the-Loop (HITL) fallback. Built with FastAPI, PostgreSQL + pgvector, Streamlit, and **100% free cloud APIs**: Groq (chat + judge), Google Gemini (embeddings), and Tavily (web search). No local models required.

---

## Architecture

```
User Query
    │
    ▼
┌─────────┐     ┌──────────────┐     ┌─────────────┐
│ Router  │────▶│ Vector Search│────▶│  Generator  │
│ (plans  │     └──────────────┘     │   (Groq     │
│  tools) │────▶┌──────────────┐────▶│  llama-3.1  │
│         │     │  Web Search  │     │ 8b-instant) │
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

**Models (all free tier):**

| Role | Model | Provider |
|---|---|---|
| Router / Generator | `llama-3.1-8b-instant` | Groq |
| Judge | `llama-3.3-70b-versatile` | Groq |
| Embeddings | `gemini-embedding-001` (768-dim) | Google Gemini |

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| Docker | Any recent | Runs PostgreSQL + pgvector (no local install needed) |
| Groq API key | — | Free tier, no credit card — [console.groq.com/keys](https://console.groq.com/keys) |
| Google Gemini API key | — | Free tier, no credit card — [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Tavily API key | — | Optional (enables web search) — free tier at [tavily.com](https://tavily.com) |
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
pip install -r app/requirements.txt
```

### 3. Start PostgreSQL + pgvector in Docker

```bash
docker compose up -d
```

This starts a `pgvector-db` container on port 5433 with a persistent `pgvector_data` volume. The pgvector extension and the `documents` table are created automatically on first ingest. If the container already exists from a previous run, `docker start pgvector-db` also works.

### 4. Configure environment variables

Copy the example below into a `.env` file at the project root (`agentic-ai/.env`) and paste in your keys:

```env
# PostgreSQL (matches the pgvector-db Docker container)
DB_NAME=vectordb
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5433
DB_TABLE=documents

# Groq (chat + judge, free tier) — https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here

# Google Gemini (embeddings, free tier) — https://aistudio.google.com/apikey
GOOGLE_API_KEY=your_gemini_api_key_here

# Models (all free-tier cloud models)
CHAT_MODEL=llama-3.1-8b-instant
JUDGE_MODEL=llama-3.3-70b-versatile
EMBED_MODEL=models/gemini-embedding-001
EMBED_DIM=768

# Tavily (web search, optional) — https://tavily.com
TAVILY_API_KEY=

# GitHub (optional — raises API rate limits)
GITHUB_TOKEN=

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
[Config] Chat model: Groq (llama-3.1-8b-instant)
[Config] Judge model: Groq (llama-3.3-70b-versatile)
```

---

## Running the Frontend

Run Streamlit alongside the API:

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
- Judge reasoning and retrieved sources in expandable panels
- HITL warning shown inline when the judge could not get a satisfactory answer after max retries
- API health indicator in the sidebar
- Clear chat button

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
  "characters_extracted": 46368,
  "chunks_stored": 117,
  "message": "Document ingested successfully."
}
```

> **Note on the free tier:** Gemini's free tier allows 100 embedding requests per minute. Documents that produce more than ~50 chunks are embedded in batches with an automatic pause between them, so a large ingest can take a couple of minutes — this is expected, not a hang.

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
  "context_sources": ["--- [Vector Search Result..."],
  "judge_log": ["ACCEPT: The answer is directly supported by the context."]
}
```

**`steps_taken` values explained:**

| Step | Description |
|---|---|
| `vector_search` | Searched the ingested document database |
| `web_search` | Searched the public web via Tavily |
| `github_read` | Fetched content from a GitHub repository |
| `generate` | Generated an answer using Groq llama-3.1-8b-instant |
| `judge` | Evaluated the answer using Groq llama-3.3-70b-versatile |
| `rewrite` | Query was rewritten and retried |
| `hitl` | Max retries reached — flagged for human review |

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
agentic-ai/
├── .env                        # Environment variables (not committed)
├── documents/                  # Sample documents for ingestion
│   └── NimbusCart_Policy_Handbook.docx
└── app/
    ├── api.py                  # FastAPI routes (/health, /ingest, /chat)
    ├── streamlit_app.py        # Streamlit chat frontend
    ├── main.py                 # CLI entry point (ingest / chat modes)
    ├── generate_graph.py       # Graph visualisation script
    ├── requirements.txt        # Python dependencies
    └── core/
        ├── graph.py            # LangGraph nodes and workflow (main logic)
        ├── embeddings.py       # Gemini embeddings (swap provider here)
        ├── generator.py        # Legacy generator (used by api.py)
        ├── embedder.py         # Embedding and pgvector storage
        ├── retriever.py        # Vector similarity search
        ├── ingestor.py         # PDF / DOCX text extraction
        └── chunker.py          # Text chunking
```
