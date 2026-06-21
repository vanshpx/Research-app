"""
Pydantic schemas for request and response models.
"""
from pydantic import BaseModel
from typing import List, Optional


class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    num_chunks: int


class Citation(BaseModel):
    page: int
    source: str
    snippet: str
    chunk_index: int = -1  # -1 when not available (e.g. from agent tool results)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieval_steps: int
    question: str


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
