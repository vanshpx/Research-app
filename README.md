# ResearchAI — Document Q&A Assistant

A RAG-powered assistant that lets you upload research PDFs and ask natural language questions. Answers are grounded in your documents with source citations.

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Python |
| Vector DB | Qdrant (in-memory) |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim) |
| LLM | Gemini 2.5 Flash |
| Chunking | RecursiveCharacterTextSplitter (1000/200) |
| Frontend | HTML + CSS + Vanilla JS |

## Project Structure

```
Research tool/
├── backend/
│   ├── .env                   ← Add your API keys here
│   ├── pyproject.toml
│   ├── main.py                ← FastAPI entry point
│   └── app/
│       ├── api/routes.py      ← /upload and /query endpoints
│       ├── loaders/           ← PDF text extraction
│       ├── chunkers/          ← Text splitting
│       ├── embeddings/        ← BAAI/bge-small-en-v1.5
│       ├── vectordb/          ← Qdrant store
│       ├── retrievers/        ← ReAct retrieval agent
│       ├── llm/               ← Gemini 2.5 Flash
│       ├── citation/          ← Citation builder
│       └── models/            ← Pydantic schemas
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

## Setup & Run

### 1. Add your API key

Edit `backend/.env`:
```
GOOGLE_API_KEY=your_actual_key_here
```

### 2. Activate virtual environment

```powershell
cd backend
.venv\Scripts\activate
```

### 3. Start the backend

```powershell
python main.py
```

The server starts at **http://localhost:8000**

- **Swagger UI**: http://localhost:8000/docs
- **Frontend**: http://localhost:8000/ (served automatically)

Or open `frontend/index.html` directly in your browser.

## API Endpoints

### `POST /api/upload`
Upload a PDF file for indexing.

**Request:** `multipart/form-data` with `file` field  
**Response:**
```json
{
  "success": true,
  "message": "Successfully processed 'paper.pdf'.",
  "filename": "paper.pdf",
  "num_chunks": 42
}
```

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

### `GET /api/health`
Returns server status and number of indexed chunks.

## How It Works

1. **Upload**: PDF → text extraction (PyPDF) → chunking (1000 chars / 200 overlap) → embeddings (BGE) → Qdrant
2. **Query**: Question → embed → cosine similarity search → ReAct agent (up to 3 retrieval steps) → Gemini 2.5 Flash → answer + citations
