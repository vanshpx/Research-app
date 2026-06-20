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
from app.retrievers.retriever import retrieve
from app.llm.gemini import generate_answer
from app.citation.citation_builder import build_citations
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
async def query_documents(body: QueryRequest, request: Request):
    """
    Ask a question against all uploaded documents.
    Uses ReAct retrieval agent + Gemini 2.5 Flash for answer generation.
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
        # Step 1: ReAct retrieval
        logger.info(f"Retrieving chunks for question: '{body.question}'")
        chunks, steps = retrieve(body.question, top_k=body.top_k or 5)

        if not chunks:
            return QueryResponse(
                answer="I could not find relevant information in the uploaded documents.",
                citations=[],
                retrieval_steps=steps,
                question=body.question,
            )

        # Step 2: Cross-encoder reranking
        reranker = getattr(request.app.state, "reranker", None)
        if reranker is not None:
            logger.info(f"Reranking {len(chunks)} chunks with cross-encoder...")
            chunks = reranker.rerank(body.question, chunks, top_k=body.top_k or 5)
            logger.info(f"Reranked down to {len(chunks)} chunks.")
        else:
            logger.warning("Reranker not available; skipping reranking step.")

        # Step 3: Generate answer with Gemini
        logger.info(f"Generating answer using {len(chunks)} chunks...")
        answer = generate_answer(body.question, chunks)

        # Step 3: Build citations
        citations = build_citations(chunks)

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieval_steps=steps,
            question=body.question,
        )

    except ValueError as e:
        # API key not configured
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
