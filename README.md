# RAG Demo — Agentic Retrieval-Augmented Generation

[![CI](https://github.com/jay0singh/agentic-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/jay0singh/agentic-ai/actions/workflows/ci.yml)

A LangGraph-powered RAG system with multi-tool routing, an LLM-as-judge evaluation loop, and a Human-in-the-Loop (HITL) fallback. Built with FastAPI, PostgreSQL + pgvector, Streamlit, and **100% free cloud APIs**: Groq (chat + judge), Google Gemini (embeddings), and Tavily (web search). No local models required.

**Highlights**

- **Streaming answers** — tokens render live in the UI via an SSE endpoint (`/chat/stream`)
- **Persistent conversations** — full transcripts live in Postgres: refresh the page and your chat is still there, switch between past conversations in the sidebar
- **Conversation memory** — follow-up questions ("does it cost anything?") are resolved against the session history before routing, surviving server restarts
- **Source citations** — every retrieved chunk carries its source filename and similarity score; answers show what they cited
- **Hybrid search** — vector similarity fused with Postgres full-text search via Reciprocal Rank Fusion, so exact-keyword queries ("section 7.4") match reliably
- **Relevance threshold** — weak vector matches are dropped instead of polluting the prompt (`RETRIEVAL_MAX_DISTANCE`)
- **LLM-as-judge loop** — a larger model grades each answer and triggers a retry with a rewritten query when it isn't grounded
- **Human-in-the-loop review queue** — questions the judge gives up on are persisted; a human answers them in the UI and the answer is taught back into the knowledge base
- **Langfuse tracing** — full traces per request with sessions, judge scores, and token usage (optional, free tier)
- **User feedback** — 👍/👎 under every answer, stored as a `user-thumbs` score on that answer's trace for quality analysis
- **Tested + CI** — 90+ mocked-LLM tests run on every push via GitHub Actions

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
                              (persists to review queue)
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

## Quick Start (one command, everything in Docker)

```bash
git clone <repo-url>
cd agentic-ai
cp .env.example .env     # then paste in your Groq + Gemini keys
docker compose up -d --build
```

That starts all three services: PostgreSQL + pgvector, the FastAPI backend (http://localhost:8000), and the Streamlit UI (http://localhost:8501). Stop everything with `docker compose down`.

Prefer running the Python app directly on your machine (e.g. for development with hot reload)? Follow the manual setup below — in that case only the database runs in Docker.

---

## Setup (manual, for development)

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

Copy [`.env.example`](.env.example) to `.env` at the project root and paste in your keys:

```bash
cp .env.example .env
```

Required: `GROQ_API_KEY` (chat + judge) and `GOOGLE_API_KEY` (embeddings) — both free, no credit card. Optional: `TAVILY_API_KEY` (web search), `GITHUB_TOKEN` (higher GitHub rate limits), `LANGFUSE_*` (tracing), and `RETRIEVAL_MAX_DISTANCE` to tune how aggressively weak vector matches are dropped (default `0.42`; on-topic queries typically score 0.20–0.30 cosine distance, off-topic 0.48+).

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
- Streaming chat — answers render token-by-token as they are generated
- Conversation sidebar — past conversations persist in Postgres; the session id lives in the URL, so a page refresh restores your chat, and you can switch, start new, or delete conversations
- Conversation memory — follow-up questions are resolved against the session ("does it cost anything?" after a return-policy question just works); Clear Chat starts a fresh session
- Citation captions under every answer showing which documents (and how many chunks) were used
- 👍/👎 feedback buttons under every answer (recorded as scores on the answer's Langfuse trace)
- Document upload and ingestion (PDF / DOCX / TXT / MD) from the sidebar; re-uploading a file replaces its chunks
- Web page ingestion — paste a URL in the sidebar and its text is fetched, cleaned, and stored
- Document manager in the sidebar — see every ingested document (chunk count, date) and delete with one click
- Step badges on every response showing exactly which tools ran (`🔍 vector_search`, `🌐 web_search`, `⚖️ judge`, etc.)
- Judge reasoning and retrieved sources in expandable panels
- HITL review queue above the chat — answer flagged questions ("Resolve & teach" adds the answer to the knowledge base) or dismiss them
- API health indicator in the sidebar

---

## API Endpoints

### `GET /health`
Check if the server is running.

```bash
curl http://127.0.0.1:8000/health
```

---

### `POST /ingest`
Upload a `.pdf`, `.docx`, `.txt` or `.md` document to the vector store.

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

### `POST /ingest/url`
Fetch a web page, extract its readable text (scripts, navigation and footers stripped), chunk, embed and store it. The URL is the document's source — re-ingesting it replaces its chunks.

```bash
curl -X POST http://127.0.0.1:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

---

### `GET /documents`
List ingested documents with chunk counts and ingest timestamps.

```bash
curl http://127.0.0.1:8000/documents
```

### `DELETE /documents/{filename}`
Remove all stored chunks for one document (404 if it doesn't exist).

```bash
curl -X DELETE http://127.0.0.1:8000/documents/NimbusCart_Policy_Handbook.docx
```

---

### `POST /chat`
Ask a question. The system routes to the appropriate tools, generates an answer, evaluates it with the judge, and retries if needed.

Optional request fields: `session_id` (enables conversation memory and groups Langfuse traces), `user_id`, and `top_k` (1–10, how many chunks to retrieve; default 3).

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is NimbusCart'\''s return policy?", "session_id": "my-session", "top_k": 3}'
```

Response:
```json
{
  "question": "What is NimbusCart's return policy?",
  "answer": "NimbusCart offers a 30-day return window...",
  "steps_taken": ["vector_search", "generate", "judge"],
  "context_sources": ["--- [Vector Search Result..."],
  "citations": [{"source": "NimbusCart_Policy_Handbook.docx", "distance": 0.203}],
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

### `POST /chat/stream`
Same request body as `/chat`, but the answer streams back as Server-Sent Events while it is being generated (this is what the Streamlit UI uses):

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is NimbusMarket?"}'
```

Each event is a JSON object:

| Event `type` | Meaning |
|---|---|
| `token` | Next piece of the answer (`content`) |
| `retry` | The judge rejected the draft answer; a new attempt follows (`reason`) |
| `done` | Final payload — same fields as the `/chat` response |
| `error` | Something failed; details are in the server log |

---

### Conversations

Full chat transcripts are persisted server-side (30-day retention):

- `GET /conversations` — recent conversations (title = first question, turn count, last activity)
- `GET /conversations/{session_id}` — full transcript with per-turn steps, citations and trace ids
- `DELETE /conversations/{session_id}` — remove a conversation's transcript and working memory

---

### HITL review queue

When the judge exhausts its retries, the question is persisted to a `hitl_queue` table instead of being lost:

- `GET /hitl` — list pending flagged questions (with the failed answer and the judge's reasoning)
- `POST /hitl/{id}/resolve` — body `{"answer": "...", "ingest": true}`; records the human answer and (by default) embeds the Q&A pair into the vector store so future similar questions retrieve it. The taught answer appears in the document manager as `hitl-{id}`.
- `POST /hitl/{id}/dismiss` — discard a flagged question

The Streamlit UI surfaces pending items in a "Review queue" panel above the chat.

---

## Hosting (Hugging Face Spaces, free)

The repo auto-deploys to a Hugging Face Space on every push to `dev` ([deploy workflow](.github/workflows/deploy.yml)). The Space runs a single container: the API stays private on localhost and only the Streamlit UI is exposed. The database is a free managed Postgres with pgvector (e.g. [Neon](https://neon.tech)) — set `DB_SSLMODE=require` for it.

One-time setup:

1. Create a free **Docker Space** at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Create a free [Neon](https://neon.tech) Postgres project and copy its connection details
3. In the Space **Settings → Variables and secrets**, add the keys listed in [deploy/README_hf.md](deploy/README_hf.md)
4. In this GitHub repo: **Settings → Secrets and variables → Actions** — add secret `HF_TOKEN` (a Hugging Face *write* token) and variable `HF_SPACE` (e.g. `username/agentic-ai`)
5. Push to `dev` (or run the Deploy workflow manually) — the Space builds and goes live

---

## Running the Tests

The test suite (90+ unit and API tests) runs with all LLM calls mocked — no API keys, database, or network needed:

```bash
pip install -r app/requirements-dev.txt
cd app
pytest
```

The same suite runs automatically on every push and pull request via GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

---

## Observability with Langfuse

Every `/chat` request is traced end-to-end (router → tools → generator → judge → rewrite/HITL) when Langfuse keys are set. Tracing is fully optional — if `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are unset, the app runs exactly as before with no Langfuse calls.

### Option A — Langfuse Cloud (free Hobby tier, no containers)

Sign up at [cloud.langfuse.com](https://cloud.langfuse.com), create a project, generate API keys under **Settings → API Keys**, and put them in `.env` with `LANGFUSE_BASE_URL=https://cloud.langfuse.com`.

### Option B — Self-hosted

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Open `http://localhost:3000`, create a project, and copy the generated public/secret keys into your `.env` with `LANGFUSE_BASE_URL=http://localhost:3000`. (Heavier: runs its own Postgres, ClickHouse, Redis, and MinIO containers.)

### What gets traced

- A full trace per chat call, named `rag-chat`, tagged `rag-demo`, with the trace input/output set to the user's question and final answer (plus the resolved standalone question for follow-ups)
- Sessions — passing `session_id` groups a whole conversation in Langfuse's Sessions view
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
├── .env.example                # Template — copy to .env and add your keys
├── docker-compose.yml          # Full stack: Postgres + API + Streamlit UI
├── Dockerfile                  # App image shared by the api and ui services
├── .github/workflows/ci.yml   # CI: runs the test suite on every push
├── documents/                  # Sample documents for ingestion
│   └── NimbusCart_Policy_Handbook.docx
└── app/
    ├── api.py                  # FastAPI routes (/health, /ingest, /chat, /chat/stream)
    ├── streamlit_app.py        # Streamlit chat frontend (streaming, citations)
    ├── main.py                 # CLI entry point (ingest / chat modes)
    ├── generate_graph.py       # Graph visualisation script
    ├── requirements.txt        # Python dependencies
    ├── requirements-dev.txt    # Test dependencies (pytest)
    ├── tests/                  # Unit + API tests, all LLM calls mocked
    └── core/
        ├── graph.py            # LangGraph nodes and workflow (main logic)
        ├── hitl.py             # Human-in-the-loop review queue storage
        ├── memory.py           # Per-session conversation memory
        ├── embeddings.py       # Gemini embeddings (swap provider here)
        ├── generator.py        # Legacy generator (used by api.py)
        ├── embedder.py         # Embedding and pgvector storage
        ├── retriever.py        # Hybrid search: vector + full-text, RRF-fused
        ├── ingestor.py         # PDF / DOCX text extraction
        └── chunker.py          # Text chunking
```
