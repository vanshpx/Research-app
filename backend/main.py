"""
Main FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.routes import router

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-load embedding model to avoid cold start on first request."""
    logger.info("Starting Research Q&A Assistant...")
    try:
        from app.embeddings.embedder import get_model
        get_model()
        logger.info("Embedding model pre-loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not pre-load embedding model: {e}")
    yield
    logger.info("Shutting down Research Q&A Assistant.")


app = FastAPI(
    title="Research Document Q&A Assistant",
    description=(
        "Upload research PDFs and ask natural language questions. "
        "Get contextually grounded answers with citations using RAG + Gemini 2.5 Flash."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from the frontend (opened directly or via local server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))

    @app.get("/{filename}", include_in_schema=False)
    async def serve_static(filename: str):
        file_path = frontend_dir / filename
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
