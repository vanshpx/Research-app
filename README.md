# ResearchAI — Document Q&A Assistant

A RAG-powered research assistant that lets you upload PDF papers and ask natural language questions. Answers are grounded in your documents with source citations, powered by a **LangGraph ReAct agent** that iteratively retrieves using hybrid search (dense + BM25 + RRF) and re-ranks with a **Cross-Encoder** before passing context to Gemini 2.5 Flash.

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Python ≥ 3.10 |
| Vector DB | Qdrant (in-memory) |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim) |
| Sparse Retrieval | BM25 (rank-bm25) |
| Rank Fusion | Reciprocal Rank Fusion (RRF) |
| Re-ranking | BAAI/bge-reranker-base (Cross-Encoder) |
| Agent Framework | LangGraph (ReAct loop, up to 3 tool calls) |
| LLM | Gemini 2.5 Flash (via LangChain wrapper) |
| Chunking | RecursiveCharacterTextSplitter (1000 chars / 200 overlap) |
| Frontend | HTML + Vanilla CSS + Vanilla JS |

---

## Project Structure

```
Research tool/
├── README.md
├── .gitignore
│
├── backend/                              ← Python FastAPI server
│   ├── .env                              ← API keys (not committed)
│   ├── main.py                           ← FastAPI entry point & lifespan startup
│   ├── pyproject.toml                    ← Project metadata & dependencies
│   ├── uv.lock                           ← uv lockfile
│   └── app/
│       ├── api/
│       │   └── routes.py                 ← /upload, /query, /health endpoints
│       ├── chunkers/
│       │   └── text_splitter.py          ← RecursiveCharacterTextSplitter wrapper
│       ├── citation/
│       │   └── citation_builder.py       ← Builds source citations from chunks
│       ├── embeddings/
│       │   └── embedder.py               ← BGE embedding model (lazy singleton)
│       ├── llm/
│       │   └── gemini.py                 ← Gemini 2.5 Flash client (native SDK)
│       ├── loaders/
│       │   └── pdf_loader.py             ← PDF text extraction via PyPDF
│       ├── models/
│       │   └── schemas.py                ← Pydantic request/response schemas
│       ├── retrievers/
│       │   ├── retriever.py              ← Dense + BM25 + RRF pipeline; rebuild_bm25()
│       │   ├── bm25_retriever.py         ← BM25 sparse retriever (rank-bm25)
│       │   ├── rrf.py                    ← Reciprocal Rank Fusion combiner
│       │   ├── cross_encoder_reranker.py ← BAAI/bge-reranker-base Cross-Encoder
│       │   └── retriver_agent/           ← LangGraph ReAct agent package
│       │       ├── agent.py              ← AgentState + build_agent_node()
│       │       ├── graph_builder.py      ← Assembles StateGraph with ToolNode
│       │       ├── retriever_tool.py     ← @tool: Dense→BM25→RRF→CrossEncoder
│       │       └── react_agent.py        ← Public API: run_agent(question)
│       ├── utils/
│       └── vectordb/
│           └── qdrant_store.py           ← Qdrant in-memory vector store
│
├── frontend/                             ← Static single-page UI
│   ├── index.html                        ← App shell & markup
│   ├── style.css                         ← Dark-mode styles & layout
│   └── app.js                            ← Upload pipeline animation, chat, citations
│
└── files/                                ← Reference documents
    ├── Product Requirement Document.pdf
    └── Technical Requirement Document.pdf
```

---

## Setup & Run

### Prerequisites

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) **or** pip
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/vanshpx/Research-app.git
cd "Research tool"
```

### 2. Add your API key

Create `backend/.env`:
```env
GOOGLE_API_KEY=your_actual_key_here
```

### 3. Install dependencies

**With uv (recommended):**
```powershell
cd backend
uv sync
.venv\Scripts\activate
```

**With pip:**
```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Start the server

```powershell
uv run main.py
```

| URL | Description |
|---|---|
| http://localhost:8000/ | Frontend UI (served automatically) |
| http://localhost:8000/docs | Swagger / OpenAPI UI |
| http://localhost:8000/api/health | Health check + chunks indexed count |

---

## How It Works

### Upload flow

```
PDF file
  → PyPDF          — extract raw text pages
  → TextSplitter   — chunk into 1000-char segments (200 overlap)
  → BGE embedder   — encode each chunk → 384-dim vector
  → Qdrant         — store vectors + metadata in-memory
  → rebuild_bm25() — rebuild BM25 index from full corpus
```

The frontend shows a **3-stage pipeline animation** during upload:
`CHUNKING → CONVERTING TO VECTOR → INDEXING`

### Query flow

```
User question
  → LangGraph ReAct agent (up to 3 reasoning rounds)
        │
        ├─ Thought: what to search for
        ├─ Action: call `retrieve` tool
        │     ├─ Dense search  (BGE + Qdrant cosine similarity)
        │     ├─ Sparse search (BM25 keyword)
        │     └─ RRF fusion   → top-10 candidates
        │     └─ CrossEncoder re-rank → top-5 chunks
        ├─ Observation: review chunks
        └─ Repeat or produce final answer
  → Gemini 2.5 Flash — synthesise answer with in-line citations
  → answer + source citations returned to frontend
```

### Startup pre-loading

On server start, three heavy resources are pre-loaded before the first request:

| # | Resource | Why eager |
|---|---|---|
| 1 | BGE embedding model | Needed on every query |
| 2 | LangGraph compiled graph + Gemini LLM | Graph compilation has overhead |
| 3 | CrossEncoder (bge-reranker-base) | ~440 MB model load takes 2–5 s |

BM25 index is **not** pre-loaded — it's built (or rebuilt) immediately after each PDF upload, since it depends on the corpus.

---

## API Endpoints

### `POST /api/upload`
Upload a PDF file for indexing.

**Request:** `multipart/form-data` with a `file` field

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed 'paper.pdf'.",
  "filename": "paper.pdf",
  "num_chunks": 42
}
```

---

### `POST /api/query`
Ask a question about uploaded documents.

**Request:**
```json
{ "question": "What is the main contribution?", "top_k": 5 }
```

**Response:**
```json
{
  "answer": "The main contribution is...",
  "citations": [
    { "source": "paper.pdf", "page": 3, "snippet": "...", "chunk_index": -1 }
  ],
  "retrieval_steps": 0,
  "question": "What is the main contribution?"
}
```

> `retrieval_steps` is always `0` — the agent manages its own internal iteration count.

---

### `GET /api/health`
Returns server status and the total number of indexed chunks.

```json
{ "status": "ok", "documents_indexed": 106 }
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ Yes | Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `QDRANT_HOST` | ❌ No | Qdrant host (defaults to in-memory mode) |
| `QDRANT_PORT` | ❌ No | Qdrant port (default: 6333) |

---

## Model Cache Behaviour

| Resource | Cached where | Rebuilt when |
|---|---|---|
| BGE embedding model | `embedder.py` module global | Never (process lifetime) |
| LangGraph compiled graph | `react_agent._graph` | Never |
| CrossEncoder model | `retriever_tool._reranker` | Never |
| Qdrant client | `qdrant_store._client` | Never |
| BM25 index | `retriever._bm25_cache` | Every PDF upload |
| Qdrant vector data | In-memory only | Lost on server restart |

---

## Notes

- **Qdrant is in-memory** — all uploaded documents are lost when the server restarts. Re-upload your PDFs after each restart.
- The **CrossEncoder re-ranker** scores every (query, chunk) pair independently, which is slower than a bi-encoder but significantly more accurate for relevance ranking.
- The **LangGraph ReAct agent** can call the retrieve tool up to 3 times per query, varying its search query each round to collect complementary chunks before producing an answer.
