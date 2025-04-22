import logging
import os
import uuid
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from rag_pipeline import RAGPipeline, RAGResponse
from indexing import Indexer, ensure_pgvector_setup # Import setup function
from data_processing import DataLoader, TextProcessor # Needed for ingestion
from utils import setup_logging, PostgresConnectionPool, get_psycopg2_dsn # Optional pool import

# --- App Setup ---
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced RAG System API (Gemini+PGVector)",
    description="API for querying a RAG system with SQL routing, using Google Gemini and PGVector.",
    version="2.0.0",
)

# --- CORS Configuration ---
origins = [
    "http://localhost:5173", # Default Vite port for frontend
    "http://127.0.0.1:5173",
    "http://localhost:3000", # Default Create React App port
    "http://127.0.0.1:3000",
    # Add the origin your frontend actually runs on if different
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# --- Global State / Singletons ---
# Use context managers or proper dependency injection frameworks for complex apps
indexer_instance: Indexer | None = None
rag_pipeline_instance: RAGPipeline | None = None

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    global indexer_instance, rag_pipeline_instance
    logger.info("API Startup: Initializing resources...")
    try:
        # 1. Ensure PGVector Extension Exists (run before Indexer init)
        ensure_pgvector_setup()

        # 2. Initialize Indexer (connects to PGVector)
        indexer_instance = Indexer()

        # 3. Initialize RAG Pipeline
        rag_pipeline_instance = RAGPipeline(indexer=indexer_instance)

        # 4. Optional: Initialize direct DB pool if needed elsewhere
        # PostgresConnectionPool.initialize(get_psycopg2_dsn(settings.postgres_url))

        logger.info("API Startup: RAG components initialized successfully.")
    except Exception as e:
        logger.critical(f"API Startup FAILED: Could not initialize RAG components: {e}", exc_info=True)
        # Prevent the app from starting fully if core components fail
        indexer_instance = None
        rag_pipeline_instance = None
        # Depending on deployment, might want to raise the exception
        # raise RuntimeError("Failed to initialize core RAG components") from e

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("API Shutdown: Cleaning up resources...")
    # Optional: Close direct DB pool if used
    # PostgresConnectionPool.closeall()
    logger.info("API Shutdown: Cleanup complete.")


# --- Dependencies ---
def get_rag_pipeline() -> RAGPipeline:
    """Dependency injector for the RAG pipeline."""
    if rag_pipeline_instance is None:
         logger.error("RAG Pipeline dependency requested, but instance is not available.")
         raise HTTPException(status_code=503, detail="RAG service is unavailable due to initialization failure.")
    return rag_pipeline_instance

def get_indexer() -> Indexer:
     """Dependency injector for the Indexer."""
     if indexer_instance is None:
         logger.error("Indexer dependency requested, but instance is not available.")
         raise HTTPException(status_code=503, detail="Indexing service is unavailable due to initialization failure.")
     return indexer_instance

# --- API Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The user's query.")

class IngestResponse(BaseModel):
    status: str
    message: str
    filename: Optional[str] = None
    chunks_ingested: Optional[int] = None


# --- Background Task for Ingestion ---
def run_file_ingestion(temp_path: str, filename: str, indexer: Indexer):
    """Function to run file ingestion (PDF, DOCX) in the background."""
    logger.info(f"[Background] Starting ingestion for: {filename}")
    docs = []
    file_ext = filename.lower().split('.')[-1]
    try:
        # --- Detect file type and load ---
        if file_ext == "pdf":
            logger.debug(f"[Background] Loading PDF: {filename}")
            docs = DataLoader.load_pdf(temp_path) # Use the correct loader method
        elif file_ext == "docx":
            logger.debug(f"[Background] Loading DOCX: {filename}")
            docs = DataLoader.load_docx(temp_path) # Use the new loader method
        else:
            logger.warning(f"[Background] Unsupported file type received for ingestion: {filename}. Skipping.")
            return
        # --- End file type detection ---

        if not docs:
            logger.warning(f"[Background] No documents loaded from {filename}. Skipping.")
            return

        processor = TextProcessor()
        splits = processor.split_documents(docs)
        if not splits:
            logger.warning(f"[Background] No chunks created for {filename}. Skipping.")
            return

        doc_metadata = {"source": filename, "uploaded_via": "API"}
        for split in splits:
            split.metadata.update(doc_metadata)

        indexer.index_documents(splits)
        logger.info(f"[Background] Successfully ingested {len(splits)} chunks from {filename}.")

    except Exception as e:
        logger.error(f"[Background] Ingestion failed for {filename}: {e}", exc_info=True)
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"[Background] Removed temporary file: {temp_path}")
        except OSError as e:
            logger.error(f"[Background] Error removing temporary file {temp_path}: {e}")


# --- API Endpoints ---
@app.get("/", tags=["Status"])
async def get_status():
    """Check the status of the API and its configuration."""
    # ... (Keep existing status logic, update model names etc.) ...
    is_ok = rag_pipeline_instance is not None and indexer_instance is not None
    return {
        "status": "ok" if is_ok else "error",
        "message": "RAG API is running." if is_ok else "RAG Service initialization failed.",
        "llm_model": settings.llm_model_name,
        "embedding_model": settings.embedding_model_name,
        "reranker_enabled": settings.use_cohere_rerank,
        "vector_store": "PGVector"
        }

@app.post("/query", response_model=RAGResponse, tags=["RAG Query"])
async def process_query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline) # Inject the pipeline
):
    """
    Receive query, route to RAG or SQL via pipeline, return answer/sources.
    """
    logger.info(f"Received query via API: '{request.query[:50]}...'")
    try:
        # The routing logic is now inside pipeline.query()
        result = pipeline.query(request.query)
        logger.info(f"Sending response for query: '{request.query[:50]}...'")
        return result
    except HTTPException:
         raise # Re-raise HTTP exceptions directly
    except Exception as e:
        logger.error(f"Unexpected error processing query '{request.query[:50]}...': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred.")


@app.post("/ingest_file", response_model=IngestResponse, tags=["Indexing"], status_code=202) # Renamed endpoint
async def ingest_file_endpoint( # Renamed function
    background_tasks: BackgroundTasks,
    # Allow multiple file types (adjust description if needed)
    file: UploadFile = File(..., description="Upload a PDF or DOCX file for ingestion."),
    indexer: Indexer = Depends(get_indexer)
):
    """
    Accepts PDF or DOCX file uploads and triggers background ingestion.
    """
    # Basic validation for allowed extensions
    allowed_extensions = {".pdf", ".docx"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
         raise HTTPException(
             status_code=400,
             detail=f"Invalid file type. Allowed types are: {', '.join(allowed_extensions)}"
        )

    logger.info(f"Received request to ingest file: {file.filename}")
    temp_path = None
    try:
        # Save uploaded file temporarily (same logic as before)
        upload_dir = "/tmp" # Consider making configurable via settings
        os.makedirs(upload_dir, exist_ok=True)
        temp_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{file.filename}")
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"Saved uploaded file to temporary path: {temp_path}")

        # --- Use the RENAMED background task ---
        background_tasks.add_task(run_file_ingestion, temp_path, file.filename, indexer)
        # --- End Use the RENAMED background task ---

        logger.info(f"Scheduled background ingestion for: {file.filename}")
        return IngestResponse(
            status="scheduled",
            message=f"Ingestion scheduled for {file.filename}. Processing will happen in the background.",
            filename=file.filename
        )

    except HTTPException:
         raise # Re-raise validation errors
    except Exception as e:
        logger.error(f"Failed to schedule file ingestion for {file.filename}: {e}", exc_info=True)
        if temp_path and os.path.exists(temp_path):
             try: os.remove(temp_path)
             except OSError: logger.error(f"Failed to clean up temp file during error: {temp_path}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule ingestion: {str(e)}")
