# ResearchAI — Document Q&A Assistant

A Self-RAG research assistant that lets you upload PDF papers and ask natural language questions. Answers are **grounded in your documents with source citations**, powered by a **LangGraph Phase 5 Self-RAG pipeline** — hybrid search (dense + BM25 + RRF) → Cross-Encoder re-rank → answer generation → Self-RAG verification loop → regeneration if needed. LLM: **Groq `qwen/qwen3-32b`**.

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
| Self-RAG Verifier | LangGraph Phase 5 (verify → regenerate loop, up to 3 retries) |
| LLM | Groq `qwen/qwen3-32b` (via LangChain ChatGroq) |
| Agent Tools | `retrieve` · `tavily_search` · `calculator` |
| Chunking | RecursiveCharacterTextSplitter (1000 chars / 200 overlap) |
| Frontend | HTML + Vanilla CSS + Vanilla JS |

---

## Project Structure

```
Research tool/
├── README.md
├── .gitignore
│
├── backend/                                    ← Python FastAPI server
│   ├── .env                                    ← API keys (not committed)
│   ├── main.py                                 ← FastAPI entry point & lifespan startup
│   ├── pyproject.toml                          ← Project metadata & dependencies
│   ├── uv.lock                                 ← uv lockfile
│   └── app/
│       ├── api/
│       │   └── routes.py                       ← /upload, /query, /health endpoints
│       ├── chunkers/
│       │   └── text_splitter.py                ← RecursiveCharacterTextSplitter wrapper
│       ├── embeddings/
│       │   └── embedder.py                     ← BGE embedding model (lazy singleton)
│       ├── llm/
│       │   └── gemini.py                       ← Groq ChatGroq client wrapper
│       ├── loaders/
│       │   └── pdf_loader.py                   ← PDF text extraction via PyPDF
│       ├── models/
│       │   └── schemas.py                      ← Pydantic request/response schemas
│       ├── tools/                              ← Agent tool implementations
│       │   ├── calculator_tool.py              ← Safe SymPy math evaluator
│       │   └── tavily_search_tool.py           ← Tavily web search tool
│       ├── retrievers/
│       │   ├── retriever.py                    ← Dense + BM25 + RRF pipeline; rebuild_bm25()
│       │   ├── bm25_retriever.py               ← BM25 sparse retriever (rank-bm25)
│       │   ├── rrf.py                          ← Reciprocal Rank Fusion combiner
│       │   ├── cross_encoder_reranker.py       ← BAAI/bge-reranker-base Cross-Encoder
│       │   ├── graph_builder.py                ← Phase 4 + Phase 5 LangGraph assembly
│       │   ├── self_verfier/                   ← Phase 5 Self-RAG verification package
│       │   │   ├── agent_state.py              ← Phase 5 AgentState TypedDict
│       │   │   ├── schemas.py                  ← VerificationResult + Verdict enum
│       │   │   ├── answer_generator.py         ← generate_node + regenerate_node factories
│       │   │   ├── generation_prompt.py        ← Generation & regeneration prompts
│       │   │   ├── verifier.py                 ← verify_node factory (structured LLM output)
│       │   │   └── verifier_prompt.py          ← Verifier system & user prompts
│       │   └── retriver_agent/                 ← Phase 4 ReAct agent package
│       │       ├── agent.py                    ← AgentState + REACT_SYSTEM_PROMPT + build_agent_node()
│       │       ├── retriever_tool.py           ← @tool: Dense→BM25→RRF→CrossEncoder + TOOL_REGISTRY
│       │       └── react_agent.py              ← Public API: run_agent(question)
│       ├── utils/
│       └── vectordb/
│           └── qdrant_store.py                 ← Qdrant in-memory vector store
│
├── frontend/                                   ← Static single-page UI
│   ├── index.html                              ← App shell & markup
│   ├── style.css                               ← Dark-mode styles & layout
│   └── app.js                                  ← Upload pipeline animation, chat, citations
│
└── files/                                      ← Reference documents
    ├── Product Requirement Document.pdf
    └── Technical Requirement Document.pdf
```

---

## Setup & Run

### Prerequisites

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) **or** pip
- A [Groq API key](https://console.groq.com/)
- A [Tavily API key](https://app.tavily.com/) (for web search tool)

### 1. Clone the repository

```bash
git clone https://github.com/vanshpx/Research-app.git
cd "Research tool"
```

### 2. Add your API keys

Create `backend/.env`:
```env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
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

## Complete Call-Flow Diagram

The diagram below shows **every function call, in order, from every module** — for both the PDF upload path and the query path.

### Server Startup

```mermaid
flowchart TD
    A([Server starts\nmain.py]) --> B["lifespan()  —  main.py"]

    B --> C["get_model()  —  embedder.py\nLoads BAAI/bge-small-en-v1.5 into memory"]
    B --> D["_get_graph()  —  react_agent.py\nLazy singleton — runs once"]

    D --> D1["build_graph(llm)  —  graph_builder.py\nPhase 4: binds tools, wires agent↔tool loop"]
    D1 --> D1a["llm.bind_tools(TOOL_REGISTRY)\n[retrieve, tavily_search, calculator]"]
    D1 --> D1b["build_agent_node(llm_with_tools)  —  agent.py"]
    D1 --> D1c["ToolNode(TOOL_REGISTRY)  —  langgraph"]
    D1 --> D1d["graph.compile()  →  phase4_graph"]

    D --> D2["build_phase5_graph(llm, phase4_graph)  —  graph_builder.py\nPhase 5: wraps Phase 4 + Self-RAG loop"]
    D2 --> D2a["_build_phase4_retrieve_node(phase4_graph)"]
    D2 --> D2b["build_generate_node(llm)  —  answer_generator.py"]
    D2 --> D2c["build_verifier_node(llm)  —  verifier.py\nllm.with_structured_output(VerificationResult)"]
    D2 --> D2d["build_regenerate_node(llm)  —  answer_generator.py"]
    D2 --> D2e["graph.compile()  →  phase5_graph  ✅ cached"]

    B --> E["_get_reranker()  —  retriever_tool.py\nLoads BAAI/bge-reranker-base CrossEncoder"]

    style A fill:#1a1a2e,color:#e0e0ff
    style B fill:#16213e,color:#e0e0ff
    style C fill:#0f3460,color:#e0e0ff
    style D fill:#0f3460,color:#e0e0ff
    style E fill:#0f3460,color:#e0e0ff
    style D1 fill:#533483,color:#e0e0ff
    style D2 fill:#533483,color:#e0e0ff
```

---

### PDF Upload Flow

```mermaid
flowchart TD
    U([POST /api/upload\nroutes.py]) --> U1["upload_pdf()  —  routes.py\nReceives multipart/form-data PDF"]

    U1 --> U2["pdf_loader.load_pdf()  —  pdf_loader.py\nExtracts raw text pages via PyPDF\n──── SSE event: CHUNKING ────"]

    U2 --> U3["text_splitter.split_text()  —  text_splitter.py\nRecursiveCharacterTextSplitter\n1000 chars · 200 overlap"]

    U3 --> U4["embed_texts(chunks)  —  embedder.py\nget_model() → BGE model\nEncodes each chunk → 384-dim vector\n──── SSE event: CONVERTING TO VECTOR ────"]

    U4 --> U5["store_chunks(chunks, embeddings)  —  qdrant_store.py\nUpserts vectors + metadata into Qdrant in-memory"]

    U5 --> U6["rebuild_bm25(corpus)  —  retriever.py\nBuilds BM25Okapi index from full text corpus\n──── SSE event: INDEXING ────"]

    U6 --> U7([UploadResponse\nsuccess · filename · num_chunks])

    style U fill:#1a1a2e,color:#e0e0ff
    style U1 fill:#16213e,color:#e0e0ff
    style U2 fill:#0f3460,color:#e0e0ff
    style U3 fill:#0f3460,color:#e0e0ff
    style U4 fill:#0f3460,color:#e0e0ff
    style U5 fill:#533483,color:#e0e0ff
    style U6 fill:#533483,color:#e0e0ff
    style U7 fill:#1a1a2e,color:#e0e0ff
```

---

### Query Flow — Full End-to-End

```mermaid
flowchart TD
    Q([POST /api/query\nroutes.py]) --> Q1["query_endpoint()  —  routes.py"]
    Q1 --> Q2["run_agent(question)  —  react_agent.py\nBuilds Phase 5 initial state dict"]

    Q2 --> P5(["Phase 5 Self-RAG Graph\ngraph_builder.py · build_phase5_graph()"])

    P5 --> N1["extract_question_node()  —  graph_builder.py\nPulls question from HumanMessage into state"]

    N1 --> N2["phase4_retrieve_node()  —  graph_builder.py\nInvokes Phase 4 graph as a black-box"]

    N2 --> P4(["Phase 4 ReAct Agent\ngraph_builder.py · build_graph()"])

    P4 --> A1["agent_node()  —  agent.py\n1. Inject REACT_SYSTEM_PROMPT if first call\n2. Count tool_call_rounds\n3. If rounds ≥ MAX_STEPS → force-answer message\n4. llm_with_tools.invoke(messages)  →  AIMessage"]

    A1 -->|"AIMessage has tool_calls"| TC["should_continue()  —  graph_builder.py\n→ tool_node"]
    A1 -->|"Plain text answer"| P4END(["Phase 4 END\nreturns phase4_output"])

    TC --> TN["ToolNode  —  langgraph\nDispatches to the called tool by name"]

    TN -->|"tool = retrieve"| RT["retrieve(query)  —  retriever_tool.py\nCalls get_retriever()"]
    RT --> RT1["DenseRetriever.retrieve()  —  retriever.py\nQdrant cosine similarity search"]
    RT --> RT2["BM25Retriever.retrieve()  —  retriever.py\nBM25Okapi keyword search"]
    RT1 & RT2 --> RRF["rrf_fusion()  —  rrf.py\nReciprocal Rank Fusion → top-10"]
    RRF --> CE["_get_reranker().compress_documents()  —  cross_encoder_reranker.py\nCrossEncoder scores each chunk → top-5"]
    CE --> TM(["ToolMessage\nJSON list of chunks"])

    TN -->|"tool = tavily_search"| TV["tavily_search(query)  —  tavily_search_tool.py\nget_tavily_client().search()  →  web results\nToolMessage"]

    TN -->|"tool = calculator"| CA["calculator(expression)  —  calculator_tool.py\n_evaluate() → sympify() safe eval\nToolMessage"]

    TM & TV & CA --> A1

    P4END --> N3["generate_node()  —  answer_generator.py\nbuild_initial_generation_prompt(question, chunks)\nllm.invoke([SystemMessage, HumanMessage])\nWrites state.answer + state.previous_answers"]

    N3 --> N4["verify_node()  —  verifier.py\nbuild_verifier_user_message(question, chunks, answer)\nstructured_llm.invoke()  →  VerificationResult\nVERDICT: SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED\nWrites state.verification_result · feedback · unsupported_claims"]

    N4 --> SR{"_should_regenerate()\ngraph_builder.py"}
    SR -->|"SUPPORTED\nOR retries ≥ MAX_RETRIES"| P5END(["Phase 5 END"])
    SR -->|"Not supported\nAND retries < 3"| N5["regenerate_node()  —  answer_generator.py\nbuild_regeneration_prompt(question, chunks,\n  previous_answer, feedback, unsupported_claims)\nllm.invoke()  →  improved answer\nIncrements state.retry_count"]
    N5 --> N4

    P5END --> EX["run_agent() extracts result  —  react_agent.py\nstate.answer  →  final answer string\n_citations_from_chunks(state.retrieved_chunks)\nLogs: verdict · retry_count · answer_len · citations"]

    EX --> QR([QueryResponse\nanswer · citations · question])

    style Q fill:#1a1a2e,color:#e0e0ff
    style Q1 fill:#16213e,color:#e0e0ff
    style Q2 fill:#16213e,color:#e0e0ff
    style P5 fill:#16213e,color:#a0c4ff,stroke:#4a9eff
    style P4 fill:#16213e,color:#a0c4ff,stroke:#4a9eff
    style N1 fill:#0f3460,color:#e0e0ff
    style N2 fill:#0f3460,color:#e0e0ff
    style A1 fill:#533483,color:#e0e0ff
    style TC fill:#1e3a5f,color:#e0e0ff
    style TN fill:#1e3a5f,color:#e0e0ff
    style RT fill:#0d5c4a,color:#e0e0ff
    style RT1 fill:#0d5c4a,color:#e0e0ff
    style RT2 fill:#0d5c4a,color:#e0e0ff
    style RRF fill:#0d5c4a,color:#e0e0ff
    style CE fill:#0d5c4a,color:#e0e0ff
    style TV fill:#0d5c4a,color:#e0e0ff
    style CA fill:#0d5c4a,color:#e0e0ff
    style N3 fill:#533483,color:#e0e0ff
    style N4 fill:#7b2d8b,color:#e0e0ff
    style N5 fill:#7b2d8b,color:#e0e0ff
    style SR fill:#4a2060,color:#e0e0ff
    style EX fill:#16213e,color:#e0e0ff
    style QR fill:#1a1a2e,color:#e0e0ff
    style P4END fill:#1a1a2e,color:#e0e0ff
    style P5END fill:#1a1a2e,color:#e0e0ff
    style TM fill:#1e3a5f,color:#e0e0ff
```

---

## How It Works

### Upload flow

```
PDF file
  → pdf_loader.load_pdf()   — extract raw text pages (PyPDF)
  → text_splitter.split()   — chunk into 1000-char segments (200 overlap)
  → embedder.embed_texts()  — encode each chunk → 384-dim BGE vector
  → qdrant_store.store()    — upsert vectors + metadata into Qdrant
  → retriever.rebuild_bm25() — rebuild BM25 index from full corpus
```

The frontend shows a **3-stage pipeline animation** during upload:
`CHUNKING → CONVERTING TO VECTOR → INDEXING`

### Query flow (summary)

```
User question
  ─── Phase 4: ReAct Retrieval (up to 3 tool rounds) ────────────────
  │   agent_node  →  [retrieve | tavily_search | calculator]  →  loop
  │        retrieve: Dense (Qdrant) + BM25 + RRF → CrossEncoder top-5
  │        tavily_search: live web results
  │        calculator: safe SymPy math evaluation
  │
  ─── Phase 5: Self-RAG Verification (up to 3 retries) ──────────────
  │   generate_node   →  first answer from retrieved chunks
  │   verify_node     →  SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED
  │   regenerate_node →  rewrite using verifier feedback  (if needed)
  │   loop back to verify_node until SUPPORTED or max retries
  │
  → answer + source citations returned to frontend
```

### Agent tools

| Tool | Module | When the agent uses it |
|---|---|---|
| `retrieve` | `retriever_tool.py` | Questions about uploaded PDFs (dense + BM25 + RRF + CrossEncoder) |
| `tavily_search` | `tools/tavily_search_tool.py` | Recent events, live data, anything outside the corpus |
| `calculator` | `tools/calculator_tool.py` | Any arithmetic, statistics, or algebraic computation |

### Startup pre-loading

On server start, three heavy resources are pre-loaded before the first request:

| # | Resource | Module | Why eager |
|---|---|---|---|
| 1 | BGE embedding model | `embedder.py` | Needed on every upload & query |
| 2 | Phase 4 + Phase 5 compiled graphs + Groq LLM | `react_agent.py` | Graph compilation has overhead |
| 3 | CrossEncoder (bge-reranker-base) | `retriever_tool.py` | ~440 MB model load takes 2–5 s |

BM25 index is **not** pre-loaded — it is rebuilt immediately after each PDF upload since it depends on the corpus.

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
    { "source": "paper.pdf", "page": 3, "snippet": "..." }
  ],
  "retrieval_steps": 0,
  "question": "What is the main contribution?"
}
```

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
| `GROQ_API_KEY` | ✅ Yes | Groq API key from [console.groq.com](https://console.groq.com/) |
| `TAVILY_API_KEY` | ✅ Yes | Tavily key from [app.tavily.com](https://app.tavily.com/) |
| `QDRANT_HOST` | ❌ No | Qdrant host (defaults to in-memory mode) |
| `QDRANT_PORT` | ❌ No | Qdrant port (default: 6333) |

---

## Model Cache Behaviour

| Resource | Cached where | Rebuilt when |
|---|---|---|
| BGE embedding model | `embedder.py` module global | Never (process lifetime) |
| Phase 4 compiled graph | `react_agent._graph` (via Phase 5) | Never |
| Phase 5 compiled graph | `react_agent._graph` | Never |
| CrossEncoder model | `retriever_tool._reranker` | Never |
| Qdrant client | `qdrant_store._client` | Never |
| BM25 index | `retriever._bm25_cache` | Every PDF upload |
| Qdrant vector data | In-memory only | Lost on server restart |

---

## Notes

- **Qdrant is in-memory** — all uploaded documents are lost when the server restarts. Re-upload your PDFs after each restart.
- The **CrossEncoder re-ranker** scores every (query, chunk) pair independently, which is slower than a bi-encoder but significantly more accurate for relevance ranking.
- **MAX_STEPS = 3** — the agent can call tools up to 3 times per query. After that it is forced to produce a final answer from what it already collected. Adjustable in `agent.py`.
- **MAX_RETRIES = 3** — the Self-RAG verifier will attempt up to 3 regeneration cycles. Adjustable in `graph_builder.py`.
- The **`qwen/qwen3-32b`** model runs with `reasoning_effort="none"` to disable the thinking chain, ensuring reliable tool-call responses from LangGraph's ToolNode.
