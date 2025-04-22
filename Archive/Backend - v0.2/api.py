# RAG_Project/Backend/api.py

import logging
import os
import uuid
import json
import re
from datetime import datetime, timezone # Import timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, Body, Path as FastApiPath
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
# Removed asyncio import as operations are now synchronous

from config import settings # Import settings from config.py
from rag_pipeline import RAGPipeline, RAGResponse
from indexing import Indexer, ensure_pgvector_setup
from summarization import DataSummarizer
from data_processing import DataLoader, TextProcessor
from utils import setup_logging, PostgresConnectionPool, get_psycopg2_dsn

# --- App Setup ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Configuration ---
CHAT_HISTORY_DIR = settings.chat_history_dir
logger.info(f"Using chat history directory from settings: {CHAT_HISTORY_DIR}")

app = FastAPI(
    title="Enhanced RAG System API (v3.1 - Chat History - Sync)",
    description="API for querying a RAG system with SQL routing, data summarization/ingestion, and chat history management (Synchronous Operations).",
    version="3.1.1", # Updated version to reflect sync changes
)

# --- CORS Configuration ---
origins = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:3000", "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Global State / Singletons ---
indexer_instance: Optional[Indexer] = None
rag_pipeline_instance: Optional[RAGPipeline] = None
summarizer_instance: Optional[DataSummarizer] = None
title_llm_instance: Optional[ChatGoogleGenerativeAI] = None

# --- Startup and Shutdown Events ---
# Startup remains async as FastAPI supports it for setup/teardown
@app.on_event("startup")
async def startup_event():
    global indexer_instance, rag_pipeline_instance, summarizer_instance, title_llm_instance
    logger.info("API Startup: Initializing resources...")
    try:
        ensure_pgvector_setup()
        logger.info("PGVector setup check complete.")
        indexer_instance = Indexer()
        logger.info("Indexer initialized.")
        summarizer_instance = DataSummarizer() # Initialize sync summarizer
        logger.info("DataSummarizer initialized.")
        rag_pipeline_instance = RAGPipeline(indexer=indexer_instance) # Initialize sync pipeline
        logger.info("RAGPipeline initialized.")
        title_llm_instance = ChatGoogleGenerativeAI(model=settings.light_llm_model_name, temperature=0.2)
        logger.info(f"Title LLM initialized ({settings.light_llm_model_name}).")
        CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("API Startup: All components initialized successfully.")
    except Exception as e:
        logger.critical(f"API Startup FAILED: {e}", exc_info=True)
        indexer_instance, rag_pipeline_instance, summarizer_instance, title_llm_instance = None, None, None, None

# Shutdown remains async
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API Shutdown: Cleaning up resources...")
    logger.info("API Shutdown: Cleanup complete.")


# --- Dependencies ---
# (Keep dependency injectors as before)
def get_rag_pipeline() -> RAGPipeline:
    if rag_pipeline_instance is None: raise HTTPException(status_code=503, detail="RAG service unavailable.")
    return rag_pipeline_instance

def get_indexer() -> Indexer:
     if indexer_instance is None: raise HTTPException(status_code=503, detail="Indexing service unavailable.")
     return indexer_instance

def get_summarizer() -> DataSummarizer:
    if summarizer_instance is None: raise HTTPException(status_code=503, detail="Summarization service unavailable.")
    return summarizer_instance

def get_title_llm() -> ChatGoogleGenerativeAI:
    if title_llm_instance is None: raise HTTPException(status_code=503, detail="Title generation service unavailable.")
    return title_llm_instance

# --- API Models ---
# (Keep Pydantic models as defined previously: QueryRequest, IngestResponse,
#  ChatMessage, ConversationSaveRequest, ConversationSaveResponse, ConversationListItem, ConversationData)
class QueryRequest(BaseModel): query: str = Field(..., min_length=3)
class IngestResponse(BaseModel): status: str; message: str; filename: Optional[str]=None; summary_id: Optional[str]=None
class ChatMessage(BaseModel):
    sender: str = Field(..., pattern=r"^(user|ai)$"); text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sources: Optional[List[Dict[str, Any]]] = None; error: Optional[bool] = None
    @validator('timestamp', pre=True, always=True)
    def ensure_timezone(cls, v): # Keep validator
        if isinstance(v, str):
            try: v = datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                try: v = datetime.fromisoformat(v)
                except ValueError: return datetime.now(timezone.utc)
        if isinstance(v, datetime): return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)
class ConversationSaveRequest(BaseModel): conversation_id: Optional[str]=None; title: Optional[str]=Field(None, min_length=1, max_length=100); messages: List[ChatMessage]
class ConversationSaveResponse(BaseModel): conversation_id: str; title: str; message: str
class ConversationListItem(BaseModel): id: str; title: str; timestamp: datetime
class ConversationData(BaseModel): id: str; title: str; created_at: datetime; messages: List[ChatMessage]


# --- Helper Functions ---
# (Keep helper functions: clean_filename, load_conversation_from_file, save_conversation_to_file)
def clean_filename(name: str) -> str:
    name = re.sub(r'[^\w\-_\. ]', '_', name); name = name.replace(' ', '_'); return name[:100]

# --- SWITCHED TO SYNC ---
def generate_chat_title(first_prompt: str, llm: ChatGoogleGenerativeAI) -> str:
    """Generates a concise title using the light LLM (synchronously)."""
    if not first_prompt: return "Chat"
    logger.info(f"Generating title (sync): '{first_prompt[:50]}...'")
    try:
        system_prompt_content = "Generate a concise and descriptive 2-3 word title for a chat conversation that starts with the following user prompt. Focus on the main topic. Examples: 'Market Analysis', 'Trial Results', 'Ansible Setup'. Respond ONLY with the title."
        user_prompt_content = f"User Prompt:\n\"{first_prompt}\"\n\nTitle:"
        messages_for_llm = [ SystemMessage(content=system_prompt_content), HumanMessage(content=user_prompt_content) ]
        title_chain = llm | StrOutputParser()
        # Use synchronous invoke
        title = title_chain.invoke(messages_for_llm) # Pass the list
        title = title.strip().strip('"').strip("'").replace('\n', ' ').replace(':', '-')
        if not title: title = "Chat"
        title = title[:50]; logger.info(f"Generated title (sync): '{title}'"); return title
    except Exception as e: logger.error(f"Error generating title (sync): {e}"); return "Chat"

def load_conversation_from_file(file_path: Path) -> Optional[ConversationData]:
    try:
        if not file_path.is_file(): raise FileNotFoundError
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        return ConversationData(**data)
    except FileNotFoundError: logger.warning(f"File not found: {file_path}"); return None
    except Exception as e: logger.error(f"Error loading/validating {file_path}: {e}"); return None

def save_conversation_to_file(file_path: Path, data: ConversationData):
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = data.model_dump_json(indent=4)
        with open(file_path, 'w', encoding='utf-8') as f: f.write(json_data)
        logger.info(f"Saved conversation to: {file_path}")
    except Exception as e: logger.error(f"Error saving {file_path}: {e}"); raise

# --- Background Task for Summarization and Ingestion ---
# NOTE: This still uses the *synchronous* version of summarizer.summarize
def run_file_ingestion(temp_path: str, filename: str, indexer: Indexer, summarizer: DataSummarizer):
    logger.info(f"[Background] Starting summarization & ingestion for: {filename}")
    try:
        logger.info(f"[Background] Summarizing {filename} using method 'auto'...")
        # Call the synchronous summarize method
        summary_json = summarizer.summarize(temp_path, file_name_override=filename, summary_method='auto')
        if not summary_json or summary_json.get("metadata", {}).get("data_type") == "error":
            error_msg = summary_json.get("metadata", {}).get("error", "Summarization failed.")
            logger.error(f"[Background] Summarization failed for {filename}: {error_msg}")
            return
        summary_id = summary_json.get("id", str(uuid.uuid4()))
        logger.info(f"[Background] Summarization successful for {filename}. ID: {summary_id}.")
        metadata_for_doc = summary_json.get("metadata", {"file_name": filename, "error": "Metadata missing", "summary_id": summary_id})
        metadata_for_doc["summary_id"] = summary_id
        doc_to_index = Document(page_content=summary_json.get("document", ""), metadata=metadata_for_doc)
        logger.info(f"[Background] Indexing summary document for {filename} with ID {summary_id}...")
        indexer.index_documents([doc_to_index], ids=[summary_id]) # Assumes index_documents is sync
        logger.info(f"[Background] Successfully summarized and indexed {filename}.")
    except Exception as e:
        logger.error(f"[Background] Error during ingestion for {filename}: {e}", exc_info=True)
    finally: # Cleanup remains the same
        try:
            if temp_path and os.path.exists(temp_path): os.remove(temp_path); logger.debug(f"[Background] Removed temp: {temp_path}")
        except NameError: logger.warning("[Background] temp_path not defined.")
        except OSError as e: logger.error(f"[Background] Error removing temp {temp_path}: {e}")

# --- API Endpoints ---

# Status endpoint remains async for startup/shutdown compatibility
@app.get("/", tags=["Status"])
async def get_status():
    # (Keep existing /status endpoint code)
    is_ok = all([rag_pipeline_instance, indexer_instance, summarizer_instance, title_llm_instance])
    status_message = "RAG API is running." if is_ok else "RAG Service initialization failed."
    return { "status": "ok" if is_ok else "error", "message": status_message, "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": { "rag_pipeline": "initialized" if rag_pipeline_instance else "failed", "indexer": "initialized" if indexer_instance else "failed",
             "summarizer": "initialized" if summarizer_instance else "failed", "title_llm": "initialized" if title_llm_instance else "failed", },
        "config": { "main_llm_model": settings.llm_model_name, "light_llm_model": settings.light_llm_model_name, "embedding_model": settings.embedding_model_name,
             "reranker_enabled": settings.use_cohere_rerank, "collection_name": settings.collection_name, "chat_history_enabled": True,
             "chat_history_dir": str(settings.chat_history_dir.resolve()) } }

# --- SWITCHED TO SYNC ---
@app.post("/query", response_model=RAGResponse, tags=["RAG Query"])
def process_query( # Changed to sync def
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    logger.info(f"Received query (sync): '{request.query[:100]}...'")
    try:
        # Call the synchronous pipeline.query method (no await)
        result: RAGResponse = pipeline.query(request.query)
        logger.info(f"Sending response (sync): '{request.query[:100]}...'")
        return result
    except HTTPException as http_exc: raise http_exc # Re-raise known HTTP errors
    except Exception as e: # Catch unexpected errors
        logger.error(f"Unexpected error processing query (sync) '{request.query[:100]}...': {e}", exc_info=True)
        # Return a generic 500 error
        raise HTTPException(status_code=500, detail="Internal server error processing query.")

# Ingestion endpoint remains async due to UploadFile/BackgroundTasks
@app.post("/ingest_file", response_model=IngestResponse, tags=["Indexing"], status_code=202)
async def ingest_file_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    indexer: Indexer = Depends(get_indexer),
    summarizer: DataSummarizer = Depends(get_summarizer) # Now sync summarizer
):
    # (Keep existing /ingest_file endpoint code - run_file_ingestion now calls sync summarize)
    logger.info(f"Received ingest request: {file.filename}")
    temp_path_str: Optional[str] = None
    try:
        upload_dir = Path(os.getenv("TEMP_UPLOAD_DIR", "/tmp/rag_uploads")); upload_dir.mkdir(parents=True, exist_ok=True)
        unique_filename = f"{uuid.uuid4()}_{file.filename}"; temp_file_path = upload_dir / unique_filename
        temp_path_str = str(temp_file_path)
        with open(temp_file_path, "wb") as f: content = await file.read(); f.write(content)
        logger.info(f"Saved temp file: {temp_path_str}")
        background_tasks.add_task(run_file_ingestion, temp_path_str, file.filename, indexer, summarizer)
        logger.info(f"Scheduled background ingestion: {file.filename}")
        return IngestResponse(status="scheduled", message=f"Ingestion scheduled.", filename=file.filename)
    except Exception as e: logger.error(f"Ingest schedule fail {file.filename}: {e}"); raise HTTPException(status_code=500, detail="Failed ingestion.")


# --- Chat History Endpoints (Synchronous where possible) ---

@app.get("/conversations", response_model=List[ConversationListItem], tags=["Chat History"])
def list_conversations(): # Changed to sync def
    # (Keep existing /conversations GET list endpoint logic - file I/O is sync)
    conversations = []
    try:
        for file_path in settings.chat_history_dir.glob("*.json"):
            try:
                conv_data = load_conversation_from_file(file_path)
                if conv_data:
                    m_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                    conversations.append(ConversationListItem(id=conv_data.id, title=conv_data.title, timestamp=m_time))
                else: logger.warning(f"Skip load fail: {file_path.name}")
            except Exception as e: logger.error(f"List file error {file_path.name}: {e}"); continue
        conversations.sort(key=lambda x: x.timestamp, reverse=True); return conversations
    except Exception as e: logger.error(f"List conversations fail: {e}"); raise HTTPException(status_code=500, detail="Cannot list chats.")

@app.get("/conversations/{conversation_id}", response_model=ConversationData, tags=["Chat History"])
def get_conversation(conversation_id: str = FastApiPath(...)): # Changed to sync def
    # (Keep existing /conversations/{id} GET endpoint logic - file I/O is sync)
    target_file: Optional[Path] = None
    try:
         pattern = f"*_{conversation_id}_*.json"; matches = list(settings.chat_history_dir.glob(pattern))
         if not matches: raise HTTPException(status_code=404, detail="Conversation not found.")
         target_file = matches[0]
         if len(matches) > 1: logger.warning(f"Multiple files for ID {conversation_id}. Using {target_file.name}")
         conversation_data = load_conversation_from_file(target_file)
         if conversation_data is None: raise HTTPException(status_code=500, detail="Failed to load/parse file.")
         if conversation_data.id != conversation_id: raise HTTPException(status_code=500, detail="Conversation ID mismatch.")
         return conversation_data
    except HTTPException: raise
    except Exception as e: logger.error(f"Get conversation {conversation_id} error: {e}"); raise HTTPException(status_code=500, detail="Cannot get chat.")

# --- SWITCHED TO SYNC ---
@app.post("/conversations", response_model=ConversationSaveResponse, tags=["Chat History"])
def save_conversation( # Changed to sync def
    request: ConversationSaveRequest = Body(...),
    title_llm: ChatGoogleGenerativeAI = Depends(get_title_llm)
):
    """Saves/updates conversation (synchronously)."""
    logger.debug(f"Save request: ID: {request.conversation_id}")
    conversation_data: Optional[ConversationData] = None; file_path: Optional[Path] = None
    is_new_chat = not request.conversation_id; response_message = "Conversation saved."
    try:
        if is_new_chat:
            logger.info("New conversation (sync)."); new_id = str(uuid.uuid4())
            if not request.messages: raise HTTPException(status_code=400, detail="Cannot create empty chat.")
            generated_title = request.title
            if not generated_title:
                 first_user_message = next((msg for msg in request.messages if msg.sender == 'user'), None)
                 # Call synchronous helper (no await)
                 generated_title = generate_chat_title(first_user_message.text if first_user_message else "", title_llm)
            if not generated_title: generated_title = "Chat"
            created_time = datetime.now(timezone.utc)
            conversation_data = ConversationData(id=new_id, title=generated_title, created_at=created_time, messages=request.messages)
            time_str = created_time.strftime("%Y%m%d_%H%M%S"); cleaned_title_part = clean_filename(generated_title)
            filename = f"{time_str}_{new_id}_{cleaned_title_part}.json"
            file_path = settings.chat_history_dir / filename; response_message = "New conversation created."
            logger.info(f"New conversation created (sync). ID: {new_id}, Title: '{generated_title}', File: {filename}")
        else: # Update
             conv_id = request.conversation_id; logger.info(f"Update conversation (sync): {conv_id}")
             pattern = f"*_{conv_id}_*.json"; matches = list(settings.chat_history_dir.glob(pattern))
             if not matches: raise HTTPException(status_code=404, detail=f"ID {conv_id} not found.")
             file_path = matches[0]
             existing_data = load_conversation_from_file(file_path)
             if not existing_data: raise HTTPException(status_code=500, detail="Failed load existing.")
             final_title = request.title if request.title else existing_data.title
             conversation_data = ConversationData(id=existing_data.id, title=final_title, created_at=existing_data.created_at, messages=request.messages)
             response_message = "Conversation updated."
             logger.info(f"Conversation updated (sync). ID: {conv_id}, Title: '{final_title}', File: {file_path.name}")
        # Save (sync)
        save_conversation_to_file(file_path, conversation_data)
        return ConversationSaveResponse(conversation_id=conversation_data.id, title=conversation_data.title, message=response_message)
    except HTTPException as http_exc: raise http_exc
    except Exception as e: logger.error(f"Save conversation error (sync) (ID: {request.conversation_id}): {e}"); raise HTTPException(status_code=500, detail="Could not save chat.")