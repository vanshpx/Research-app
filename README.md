# ResearchAI — Document Q&A Assistant

A RAG-powered assistant that lets you upload research PDFs and ask natural language questions. Answers are grounded in your documents with source citations, powered by hybrid retrieval (dense + BM25 with Reciprocal Rank Fusion) and a ReAct agent loop.

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Python ≥ 3.10 |
| Vector DB | Qdrant (in-memory) |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim) |
| Sparse Retrieval | BM25 (rank-bm25) |
| Rank Fusion | Reciprocal Rank Fusion (RRF) |
| LLM | Gemini 2.5 Flash |
| Chunking | RecursiveCharacterTextSplitter (1000 / 200 overlap) |
| Frontend | HTML + CSS + Vanilla JS |

## Project Structure

```
Research tool/
├── README.md
├── .gitignore
│
├── backend/                          ← Python FastAPI server
│   ├── .env                          ← API keys (not committed)
│   ├── main.py                       ← FastAPI entry point & server startup
│   ├── pyproject.toml                ← Project metadata & dependencies
│   ├── requirements.txt              ← Pinned pip requirements
│   ├── uv.lock                       ← uv lockfile
│   └── app/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── routes.py             ← /upload, /query, /health endpoints
│       ├── chunkers/
│       │   ├── __init__.py
│       │   └── text_splitter.py      ← RecursiveCharacterTextSplitter wrapper
│       ├── citation/
│       │   ├── __init__.py
│       │   └── citation_builder.py   ← Builds source citations from chunks
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── embedder.py           ← BAAI/bge-small-en-v1.5 embedding model
│       ├── llm/
│       │   ├── __init__.py
│       │   └── gemini.py             ← Gemini 2.5 Flash client & prompt logic
│       ├── loaders/
│       │   ├── __init__.py
│       │   └── pdf_loader.py         ← PDF text extraction via PyPDF
│       ├── models/
│       │   ├── __init__.py
│       │   └── schemas.py            ← Pydantic request/response schemas
│       ├── retrievers/
│       │   ├── __init__.py
│       │   ├── retriever.py          ← ReAct retrieval agent (up to 3 steps)
│       │   ├── bm25_retriever.py     ← BM25 sparse retriever
│       │   └── rrf.py                ← Reciprocal Rank Fusion combiner
│       ├── utils/
│       │   └── __init__.py
│       └── vectordb/
│           ├── __init__.py
│           └── qdrant_store.py       ← Qdrant in-memory vector store
│
├── frontend/                         ← Static single-page UI
│   ├── index.html                    ← App shell & markup
│   ├── style.css                     ← Styles & layout
│   └── app.js                        ← Upload, query & rendering logic
│
└── files/                            ← Reference documents
    ├── Product Requirement Document.pdf
    └── Technical Requirement Document.pdf
```

## Setup & Run

### Prerequisites

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) **or** pip
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

---

### 1. Clone the repository

```bash
git clone https://github.com/vanshpx/Research-app.git
cd "Research app"
```

### 2. Add your API key

Edit `backend/.env`:
```env
GOOGLE_API_KEY=your_actual_key_here
```

### 3. Install dependencies & activate virtual environment

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

### 4. Start the backend

```powershell
python main.py
```

The server starts at **http://localhost:8000**

| URL | Description |
|---|---|
| http://localhost:8000/ | Frontend UI (served automatically) |
| http://localhost:8000/docs | Swagger / OpenAPI UI |
| http://localhost:8000/api/health | Health check |

> You can also open `frontend/index.html` directly in your browser (no server needed for the UI alone).

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
    { "page": 3, "source": "paper.pdf", "snippet": "...", "chunk_index": 12 }
  ],
  "retrieval_steps": 1,
  "question": "What is the main contribution?"
}
```

---

### `GET /api/health`
Returns server status and the total number of indexed chunks.

---

## How It Works

```
Upload flow:
  PDF → PyPDF (text extraction)
      → RecursiveCharacterTextSplitter (1000 chars / 200 overlap)
      → BGE embeddings (384-dim)
      → Qdrant (in-memory vector store)
      → BM25 index

Query flow:
  Question → embed query
           → Qdrant cosine similarity search  ┐
           → BM25 keyword search              ┘ → RRF fusion
           → ReAct agent (up to 3 retrieval steps)
           → Gemini 2.5 Flash
           → answer + source citations
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ Yes | Gemini API key from Google AI Studio |
| `QDRANT_HOST` | ❌ No | Qdrant host (defaults to in-memory) |
| `QDRANT_PORT` | ❌ No | Qdrant port (default: 6333) |
