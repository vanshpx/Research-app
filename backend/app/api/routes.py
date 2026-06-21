"""
API Routes — FastAPI endpoint definitions for /upload and /query.
"""
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse

from app.loaders.pdf_loader import load_pdf
from app.chunkers.text_splitter import chunk_pages
from app.embeddings.embedder import embed_texts
from app.vectordb.qdrant_store import upsert_chunks, get_document_count
from app.retrievers.retriever import rebuild_bm25
from app.retrievers.retriver_agent.react_agent import run_agent
from app.models.schemas import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and number of indexed document chunks."""
    count = get_document_count()
    return HealthResponse(status="ok", documents_indexed=count)


@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract text, chunk it, embed it, and store in Qdrant.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        filename = file.filename

        # Step 1: Load PDF
        logger.info(f"Loading PDF: {filename}")
        pages = load_pdf(tmp_path)
        if not pages:
            raise HTTPException(status_code=422, detail="Could not extract text from the PDF.")

        # Step 2: Chunk pages
        logger.info(f"Chunking {len(pages)} pages...")
        chunks = chunk_pages(pages, filename=filename)

        # Step 3: Embed chunks
        logger.info(f"Embedding {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        # Step 4: Store in Qdrant
        logger.info("Storing embeddings in Qdrant...")
        upsert_chunks(chunks, embeddings)

        # Step 5: Rebuild BM25 index with the newly added chunks
        logger.info("Rebuilding BM25 index...")
        rebuild_bm25()
        logger.info("BM25 index updated.")

        return UploadResponse(
            success=True,
            message=f"Successfully processed '{filename}'.",
            filename=filename,
            num_chunks=len(chunks),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/query", response_model=QueryResponse, tags=["Q&A"])
async def query_documents(body: QueryRequest):
    """
    Ask a question against all uploaded documents.
    Uses the LangGraph ReAct agent:
      - Iteratively calls the retrieve tool (Dense + BM25 + RRF + CrossEncoder)
      - Gemini 2.5 Flash reasons over retrieved chunks and generates the answer
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    total_chunks = get_document_count()
    if total_chunks == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents have been uploaded yet. Please upload a PDF first.",
        )

    try:
        logger.info(f"[Agent] Running ReAct agent for: '{body.question}'")
        answer, citations = run_agent(body.question)

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieval_steps=0,   # agent manages its own steps internally
            question=body.question,
        )

    except ValueError as e:
        # API key not configured
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
