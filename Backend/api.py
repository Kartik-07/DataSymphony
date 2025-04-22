# RAG_Project/MY_RAG/Backend/api.py

import logging
import os
import uuid
import json
import re
from datetime import datetime, timezone, timedelta # Import timedelta
from typing import Optional, List, Dict, Any, Annotated # Added Annotated
from pathlib import Path

# --- FastAPI and Security Imports ---
from fastapi import (
    FastAPI, HTTPException, Depends, UploadFile, File,
    BackgroundTasks, Body, Path as FastApiPath, status, APIRouter # Added APIRouter
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
# --- Updated Pydantic Import ---
from pydantic import BaseModel, Field, EmailStr # field_validator is not directly used here, but needed by models.py

# --- Langchain and Core Logic Imports ---
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# --- Local Module Imports ---
from config import settings # Import settings from config.py
# Assuming RAGPipeline & RAGResponse are defined correctly in rag_pipeline.py
from rag_pipeline import RAGPipeline, RAGResponse
from indexing import Indexer, ensure_pgvector_setup
from summarization import DataSummarizer # Assuming sync version
from data_processing import DataLoader, TextProcessor
# --- Updated utils import ---
from utils import setup_logging, save_temporary_file, read_temporary_file_content # Import new utils functions
# Import shared models from models.py
from models import ChatMessage, ConversationData, ConversationListItem

# --- NEW: Auth and Memory Manager Imports ---
from auth_manager import auth_manager, UserRegistration, UserLogin, UserData as AuthUserData
from memory_manager import memory_manager


# --- App Setup ---
# setup_logging() # Call this in main.py or app factory instead
logger = logging.getLogger(__name__)

# Use APIRouter if splitting endpoints later, otherwise direct app decorators are fine
# For simplicity here, we'll keep using app decorators as in the original code.
# If you plan to split into multiple files, use APIRouter.
# router = APIRouter() # Example if using router

app = FastAPI(
    title="Enhanced RAG System API (v4.7 - Temp File Upload)", # Updated title
    description="API for querying a RAG system with user authentication, history, SQL routing, summarization/ingestion, and temporary file uploads for queries.",
    version="4.7.0", # Updated version
)

# --- CORS Configuration ---
origins = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:3000", "http://127.0.0.1:3000",
    # Add other origins as needed
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
@app.on_event("startup")
async def startup_event():
    global indexer_instance, rag_pipeline_instance, summarizer_instance, title_llm_instance
    # Ensure logging is configured *before* logging messages
    setup_logging()
    logger.info("API Startup: Initializing resources...")
    try:
        logger.info(f"Auth directory: {settings.auth_dir}")
        logger.info(f"User history base directory: {settings.user_history_base_dir}")
        ensure_pgvector_setup()
        logger.info("PGVector setup check complete.")
        indexer_instance = Indexer()
        logger.info("Indexer initialized.")
        summarizer_instance = DataSummarizer()
        logger.info("DataSummarizer initialized.")
        rag_pipeline_instance = RAGPipeline(indexer=indexer_instance)
        logger.info("RAGPipeline initialized.")
        title_llm_instance = ChatGoogleGenerativeAI(model=settings.light_llm_model_name, temperature=0.2)
        logger.info(f"Title LLM initialized ({settings.light_llm_model_name}).")
        # --- IMPORTANT: Start the cleanup scheduler from utils ---
        # Assuming start_cleanup_scheduler is defined in utils.py
        from utils import start_cleanup_scheduler
        start_cleanup_scheduler()
        # --------------------------------------------------------
        logger.info("API Startup: All components initialized successfully.")
    except Exception as e:
        logger.critical(f"API Startup FAILED: {e}", exc_info=True)
        # Ensure instances are None on failure
        indexer_instance, rag_pipeline_instance, summarizer_instance, title_llm_instance = None, None, None, None

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API Shutdown: Cleaning up resources...")
    # Add any specific cleanup here if needed (e.g., closing DB pool from utils)
    # from utils import PostgresConnectionPool
    # PostgresConnectionPool.closeall() # Example if using the pool
    logger.info("API Shutdown: Cleanup complete.")


# --- Dependencies (Keep RAG dependencies) ---
# These functions act as injectors for the global instances
def get_rag_pipeline() -> RAGPipeline:
    if rag_pipeline_instance is None:
        logger.error("RAG pipeline accessed before initialization.")
        raise HTTPException(status_code=503, detail="RAG service unavailable.")
    return rag_pipeline_instance

def get_indexer() -> Indexer:
    if indexer_instance is None:
        logger.error("Indexer accessed before initialization.")
        raise HTTPException(status_code=503, detail="Indexing service unavailable.")
    return indexer_instance

def get_summarizer() -> DataSummarizer:
    if summarizer_instance is None:
        logger.error("Summarizer accessed before initialization.")
        raise HTTPException(status_code=503, detail="Summarization service unavailable.")
    return summarizer_instance

def get_title_llm() -> ChatGoogleGenerativeAI:
    if title_llm_instance is None:
        logger.error("Title LLM accessed before initialization.")
        raise HTTPException(status_code=503, detail="Title generation service unavailable.")
    return title_llm_instance

# --- NEW: Security Setup (JWT Tokens) ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token") # Endpoint for login

# --- NEW: Dependency for Getting Current Authenticated User ---
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> AuthUserData:
    """Dependency function to get the current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        email: str | None = payload.get("sub") # Use | None for type safety
        if email is None:
            logger.warning("Token payload missing 'sub' (email)")
            raise credentials_exception
        # Validate email format using Pydantic's EmailStr implicitly if needed,
        # but auth_manager.get_user likely handles this.
    except JWTError as e:
        logger.warning(f"JWTError decoding token: {e}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected error decoding or validating token: {e}", exc_info=True)
        raise credentials_exception

    # Fetch user using validated email
    try:
        user = auth_manager.get_user(email) # Pass the email string directly
    except Exception as getUserError: # Catch potential errors in get_user itself
        logger.error(f"Error fetching user '{email}' from auth manager: {getUserError}", exc_info=True)
        raise credentials_exception # Raise the original exception if user fetch fails

    if user is None:
        logger.warning(f"User '{email}' from valid token not found in auth storage.")
        raise credentials_exception
    return user

# --- NEW: Token Creation Helper ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Helper function to create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Use configured expiration time
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


# --- API Models (Check if any need to be moved to models.py) ---

# --- MODIFIED QueryRequest to include temp_file_id ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's query text.") # Allow shorter queries if needed
    conversation_id: Optional[str] = Field(None, description="Optional ID of the conversation for context and history update")
    temp_file_id: Optional[str] = Field(None, description="Optional ID of a temporarily uploaded file to use as context.")


class IngestResponse(BaseModel):
    status: str
    message: str
    filename: Optional[str] = None
    summary_id: Optional[str] = None

class ConversationSaveRequest(BaseModel):
    conversation_id: Optional[str] = None
    title: Optional[str] = Field(None, min_length=1, max_length=100)
    messages: List[ChatMessage] # Uses ChatMessage from models.py

class ConversationSaveResponse(BaseModel):
    conversation_id: str
    title: str
    message: str

# --- Authentication Specific API Models ---
class UserResponse(BaseModel):
    email: EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str

# --- NEW: Temporary File Upload Response Model ---
class TempUploadResponse(BaseModel):
    file_id: str
    filename: str


# --- Helper Functions ---
def clean_filename(name: str) -> str:
    """Cleans a string to be suitable for use in a filename component."""
    if not name: return "untitled"
    name = re.sub(r'[^\w\-_\. ]', '_', name) # Allow alphanumeric, hyphen, underscore, period, space
    name = name.strip().replace(' ', '_') # Trim and replace spaces
    return name[:100] # Limit length

# Title generation remains synchronous
def generate_chat_title(first_prompt: str, llm: ChatGoogleGenerativeAI) -> str:
    """Generates a concise title using the light LLM (synchronously)."""
    if not first_prompt: return "New Chat"
    logger.info(f"Generating title (sync) based on: '{first_prompt[:50]}...'")
    try:
        system_prompt = "Generate a concise, descriptive title (3-5 words) for a chat conversation starting with the user prompt below. Focus on the core topic. Examples: 'Market Analysis Q1', 'Debugging Python Script', 'Vacation Planning'. Respond ONLY with the title itself."
        messages_for_llm = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Prompt:\n\"{first_prompt}\"")
        ]
        # Ensure llm is not None before invoking
        if not llm:
            logger.error("Title generation LLM is not available.")
            return "Chat" # Fallback title
        title_chain = llm | StrOutputParser()
        title = title_chain.invoke(messages_for_llm)
        # Clean up potential markdown/quotes
        title = title.strip().strip('"\'`').replace('\n', ' ').replace(':', '-').strip()
        title = title[:60] # Limit length
        if not title: title = "Chat" # Fallback if empty after cleaning
        logger.info(f"Generated title (sync): '{title}'")
        return title
    except Exception as e:
        logger.error(f"Error generating title (sync): {e}", exc_info=True)
        return "Chat" # Fallback title on error

# --- Background Task Function (for /ingest_file) ---
def run_file_ingestion(temp_path: str, filename: str, indexer: Indexer, summarizer: DataSummarizer, user_email: Optional[str] = None):
    """Background task to summarize and index an uploaded file (for permanent storage)."""
    log_prefix = f"[Background Ingestion Task{' for user '+user_email if user_email else ''}]"
    logger.info(f"{log_prefix} Starting ingestion for: {filename}")
    summary_id = None
    try:
        if not summarizer or not indexer:
            logger.error(f"{log_prefix} Summarizer or Indexer not available. Aborting ingestion for {filename}.")
            return

        logger.info(f"{log_prefix} Summarizing {filename} using method 'auto'...")
        # Ensure temp_path exists before passing to summarizer
        if not os.path.exists(temp_path):
            logger.error(f"{log_prefix} Temporary file {temp_path} not found. Aborting summarization for {filename}.")
            return

        summary_json = summarizer.summarize(temp_path, file_name_override=filename, summary_method='auto')

        if not summary_json or summary_json.get("metadata", {}).get("data_type") == "error":
            error_msg = summary_json.get("metadata", {}).get("error", "Summarization failed, reason unknown.") if summary_json else "Summarizer returned None or empty response."
            logger.error(f"{log_prefix} Summarization failed for {filename}: {error_msg}")
            return # Stop processing if summarization failed

        summary_id = summary_json.get("id", str(uuid.uuid4()))
        logger.info(f"{log_prefix} Summarization successful for {filename}. Summary ID: {summary_id}.")

        metadata_for_doc = summary_json.get("metadata", {"file_name": filename})
        # Ensure essential metadata is present
        metadata_for_doc["summary_id"] = summary_id
        metadata_for_doc["original_filename"] = filename
        if user_email:
            metadata_for_doc["uploaded_by"] = user_email

        # Create the Document object for indexing
        doc_to_index = Document(
            page_content=summary_json.get("document", ""), # Use summary text as page content
            metadata=metadata_for_doc
        )

        logger.info(f"{log_prefix} Indexing summary document for {filename} with ID {summary_id}...")
        # Index the single summary document with its unique ID
        indexer.index_documents([doc_to_index], ids=[summary_id])
        logger.info(f"{log_prefix} Successfully summarized and indexed {filename} (Summary ID: {summary_id}).")

    except Exception as e:
        logger.error(f"{log_prefix} Error during ingestion processing for {filename}: {e}", exc_info=True)

    finally:
        # Ensure temp file (from this specific ingestion process) is removed
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"{log_prefix} Removed temp file: {temp_path}")
        except OSError as e:
            logger.error(f"{log_prefix} Error removing temp file {temp_path}: {e}")


# --- API Endpoints ---

# --- Status Endpoint (No Auth Required) ---
@app.get("/", tags=["Status"], summary="Get API Status and Configuration")
async def get_status():
    # Check if instances are initialized
    is_ok = all([rag_pipeline_instance, indexer_instance, summarizer_instance, title_llm_instance])
    status_message = "RAG API is running." if is_ok else "RAG API Service initialization failed."
    return {
        "status": "ok" if is_ok else "error",
        "message": status_message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "rag_pipeline": "initialized" if rag_pipeline_instance else "failed",
            "indexer": "initialized" if indexer_instance else "failed",
            "summarizer": "initialized" if summarizer_instance else "failed",
            "title_llm": "initialized" if title_llm_instance else "failed",
        },
        "config": {
            "main_llm_model": settings.llm_model_name,
            "light_llm_model": settings.light_llm_model_name,
            "embedding_model": settings.embedding_model_name,
            "reranker_enabled": settings.use_cohere_rerank,
            "collection_name": settings.collection_name,
            "auth_enabled": True,
            "user_history_enabled": True,
            "temp_file_uploads_enabled": True, # Added flag
        }
    }

# --- NEW Authentication Endpoints ---

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Authentication"], summary="Register a New User")
def register_new_user(user_in: UserRegistration = Body(...)):
    """Registers a new user with email and password."""
    logger.info(f"Registration attempt for email: {user_in.email}")
    existing_user = auth_manager.get_user(user_in.email)
    if existing_user:
        logger.warning(f"Registration failed: Email '{user_in.email}' already registered.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    user = auth_manager.register_user(user_in)
    if not user:
        logger.error(f"User registration failed unexpectedly for {user_in.email}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not register user due to server error.",
        )
    # Return only non-sensitive info
    return UserResponse(email=user.email)


@app.post("/token", response_model=Token, tags=["Authentication"], summary="Login and Get Access Token")
async def login_for_access_token(
    # Use Annotated correctly for dependencies within form data routes
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
):
    """Authenticates user and returns JWT access token."""
    logger.info(f"Login attempt for username (email): {form_data.username}")

    # Pass the raw username string directly to the model for validation & authentication
    try:
        user = auth_manager.authenticate_user(UserLogin(email=form_data.username, password=form_data.password))
    except ValueError as e: # Catch potential Pydantic validation error from UserLogin
        logger.warning(f"Login failed: Invalid email format for '{form_data.username}' during UserLogin creation.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, # Or 400 Bad Request
            detail="Incorrect email or password", # Keep generic for security
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user:
        # Keep error message generic for security if authentication itself fails
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # If authentication succeeds, create the token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    logger.info(f"Token issued for user: {user.email}")
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=UserResponse, tags=["Authentication"], summary="Get Current User Info")
async def read_users_me(
    # CORRECTED: Removed the redundant '= Depends(...)'
    current_user: Annotated[AuthUserData, Depends(get_current_user)]
):
    """Returns the information of the currently authenticated user."""
    logger.debug(f"Returning info for authenticated user: {current_user.email}")
    return UserResponse(email=current_user.email)


# --- MODIFIED Chat History Endpoints (Require Authentication) ---

@app.get("/conversations", response_model=List[ConversationListItem], tags=["Chat History"], summary="List User's Conversations")
def list_conversations_for_user(
    # CORRECTED: Removed the redundant '= Depends(...)'
    current_user: Annotated[AuthUserData, Depends(get_current_user)]
):
    """Lists all conversations for the currently authenticated user."""
    logger.info(f"Listing conversations for user: {current_user.email}")
    conversations = memory_manager.list_conversations(current_user.email)
    # Ensure timestamps are handled correctly if needed (e.g., timezone)
    return conversations


@app.get("/conversations/{conversation_id}", response_model=ConversationData, tags=["Chat History"], summary="Get Specific Conversation")
def get_conversation_for_user(
    # Dependency first
    current_user: Annotated[AuthUserData, Depends(get_current_user)],
    # Path parameter after
    conversation_id: str = FastApiPath(..., description="The ID of the conversation to retrieve")
):
    """Retrieves a specific conversation by ID for the authenticated user."""
    logger.info(f"Getting conversation {conversation_id} for user: {current_user.email}")
    conversation_data = memory_manager.load_conversation(current_user.email, conversation_id)
    if conversation_data is None:
        logger.warning(f"Conversation {conversation_id} not found for user {current_user.email}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return conversation_data


@app.post("/conversations", response_model=ConversationSaveResponse, tags=["Chat History"], summary="Save or Create Conversation")
def save_conversation_for_user(
    # Dependencies first
    current_user: Annotated[AuthUserData, Depends(get_current_user)],
    title_llm: Annotated[ChatGoogleGenerativeAI, Depends(get_title_llm)],
    # Body parameter after dependencies
    request: ConversationSaveRequest = Body(...)
):
    """Saves a new conversation or updates an existing one for the authenticated user."""
    user_email = current_user.email
    logger.info(f"Save conversation request for user: {user_email} (Incoming ID: {request.conversation_id})")

    is_new_chat = not request.conversation_id
    conversation_data: Optional[ConversationData] = None
    response_message: str

    try:
        if is_new_chat:
            logger.info(f"Creating new conversation for user: {user_email}")
            if not request.messages:
                logger.warning(f"Attempt to create empty chat for user: {user_email}")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot create an empty chat.")

            new_id = str(uuid.uuid4())
            generated_title = request.title # Use provided title if available
            if not generated_title:
                first_user_message = next((msg for msg in request.messages if msg.sender == 'user'), None)
                prompt_text = first_user_message.text if first_user_message else ""
                generated_title = generate_chat_title(prompt_text, title_llm)

            created_time = datetime.now(timezone.utc)
            # Ensure all timestamps in incoming messages are valid (Pydantic handles this on validation)
            conversation_data = ConversationData(
                id=new_id,
                title=generated_title,
                created_at=created_time,
                messages=request.messages
            )
            response_message = "New conversation created."
            logger.info(f"New conversation created. ID: {new_id}, Title: '{generated_title}' for user {user_email}")
        else:
            # Update existing conversation
            conv_id = request.conversation_id
            logger.info(f"Updating conversation {conv_id} for user: {user_email}")
            existing_data = memory_manager.load_conversation(user_email, conv_id)
            if not existing_data:
                logger.warning(f"Update failed: Conversation ID {conv_id} not found for user {user_email}")
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Conversation ID {conv_id} not found.")

            final_title = request.title if request.title else existing_data.title
            # Use existing created_at time, only update messages and title
            conversation_data = ConversationData(
                id=existing_data.id, # Keep original ID
                title=final_title,
                created_at=existing_data.created_at, # Keep original creation time
                messages=request.messages # Replace messages with the new list
            )
            response_message = "Conversation updated."
            logger.info(f"Conversation updated. ID: {conv_id}, Title: '{final_title}' for user {user_email}")

        # Save the new or updated conversation data
        if not memory_manager.save_conversation(user_email, conversation_data):
            logger.error(f"MemoryManager failed to save conversation {conversation_data.id} for user {user_email}.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save conversation data.")

        return ConversationSaveResponse(
            conversation_id=conversation_data.id, title=conversation_data.title, message=response_message
        )
    except HTTPException as http_exc:
        # Re-raise known HTTP exceptions
        raise http_exc
    except Exception as e:
        # Catch unexpected errors during save/create logic
        logger.error(f"Unexpected error saving conversation (ID: {request.conversation_id}) for user {user_email}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not save chat due to a server error.")


# --- NEW: Temporary File Upload Endpoint ---
@app.post("/upload_temp", response_model=TempUploadResponse, tags=["Temporary Files"], summary="Upload Temporary File for Query Context")
async def upload_temporary_file_endpoint(
    # Dependencies first
    current_user: Annotated[AuthUserData, Depends(get_current_user)],
    # File parameter after dependencies
    file: UploadFile = File(...)
):
    """
    Accepts a file upload (.txt, .pdf, .docx, .csv, .xlsx), saves it temporarily
    using the logic in utils.py, and returns a unique file ID.
    This file can be referenced in the /query endpoint via 'temp_file_id'.
    Files are automatically deleted after a configured duration (e.g., 24 hours).
    """
    user_email = current_user.email
    original_filename = file.filename if file.filename else "unknown_file"
    logger.info(f"Temporary file upload request from user {user_email} for file: {original_filename}")

    try:
        # The save_temporary_file function from utils.py handles validation and saving
        # It raises HTTPException on errors (e.g., wrong file type, save failure)
        file_id = save_temporary_file(file) # file is automatically closed within this function
        logger.info(f"Temporary file saved successfully for user {user_email}. File ID: {file_id}, Original Name: {original_filename}")
        return TempUploadResponse(file_id=file_id, filename=original_filename)
    except HTTPException as e:
        # Log and re-raise HTTP exceptions from the utility function
        logger.warning(f"HTTPException during temporary file upload for user {user_email}: {e.detail}")
        raise e
    except Exception as e:
        # Catch any other unexpected errors during upload handling
        logger.error(f"Unexpected error during temporary file upload for user {user_email}, file {original_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred during file upload.")


# --- MODIFIED RAG Query Endpoint (Handles Temp Files & History) ---
@app.post("/query", response_model=RAGResponse, tags=["RAG Query"], summary="Query RAG Pipeline with History and Optional Temp File")
# CORRECTED: Removed redundant '= Depends(...)' and applied Annotated consistently
def process_query_for_user(
    # Body parameter usually first if required
    request: QueryRequest,
    # Dependencies after required parameters
    current_user: Annotated[AuthUserData, Depends(get_current_user)],
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)] # Use Annotated
):
    """
    Processes a query through the RAG pipeline for the authenticated user.
    Loads history if conversation_id is provided.
    If temp_file_id is provided, reads the temporary file content and adds it to the query context.
    Appends the new interaction back to the specified conversation history if applicable.
    """
    user_email = current_user.email
    conversation_id = request.conversation_id
    temp_file_id = request.temp_file_id # Get the temp file ID from the request
    query_text = request.query

    log_msg = f"Received query from user {user_email}: '{query_text[:100]}...'"
    if conversation_id: log_msg += f" (Conversation ID: {conversation_id})"
    if temp_file_id: log_msg += f" (Temp File ID: {temp_file_id})"
    logger.info(log_msg)

    conversation_history: Optional[List[ChatMessage]] = None
    loaded_conv_data: Optional[ConversationData] = None
    temp_file_content: Optional[str] = None
    final_query_text: str = query_text # Start with the original query

    # --- Load Temporary File Content (if ID provided) ---
    if temp_file_id:
        logger.debug(f"Attempting to read temporary file content for ID: {temp_file_id}")
        try:
            temp_file_content = read_temporary_file_content(temp_file_id)
            logger.info(f"Successfully read content from temporary file {temp_file_id} (approx {len(temp_file_content)} chars).")
        except HTTPException as e:
            # Handle case where file is not found gracefully (e.g., expired or invalid ID)
            if e.status_code == status.HTTP_404_NOT_FOUND:
                logger.warning(f"Temporary file {temp_file_id} not found (maybe expired or invalid?). Proceeding without file content.")
                # Optionally, inform the user in the response? For now, just proceed.
            else:
                # Log other errors during file reading but proceed without content
                logger.error(f"HTTPException reading temp file {temp_file_id}: {e.detail}", exc_info=True)
            temp_file_content = None # Ensure content is None on error
        except Exception as e:
            logger.error(f"Unexpected error reading temp file {temp_file_id}: {e}", exc_info=True)
            temp_file_content = None # Ensure content is None on error

    # --- Prepend file content to the query if successfully read ---
    if temp_file_content:
        # Simple prepending strategy (adjust as needed for your RAG pipeline's prompting)
        final_query_text = (
            f"Based on the following document content:\n"
            f"--- Start Document ---\n{temp_file_content}\n--- End Document ---\n\n"
            f"User Query: {query_text}"
        )
        logger.debug(f"Prepended temporary file content to query. New query length approx: {len(final_query_text)}")


    # --- Load conversation history (if ID provided) ---
    if conversation_id:
        logger.debug(f"Attempting to load history for conversation {conversation_id} for user {user_email}")
        loaded_conv_data = memory_manager.load_conversation(user_email, conversation_id)
        if loaded_conv_data:
            conversation_history = loaded_conv_data.messages
            logger.info(f"Loaded {len(conversation_history)} messages from conversation {conversation_id} for context.")
        else:
            # If ID provided but not found, treat as new chat attempt but log warning
            logger.warning(f"Conversation {conversation_id} provided but not found for user {user_email}. Proceeding without history.")
            conversation_id = None # Ensure we don't try to save to non-existent ID later

    # --- Execute the RAG pipeline query ---
    try:
        logger.debug(f"Calling RAG pipeline for user {user_email} with final query text (length approx {len(final_query_text)}).")
        result: RAGResponse = pipeline.query(
            query_text=final_query_text, # Use the potentially modified query
            conversation_history=conversation_history # Pass loaded history (or None)
        )
        logger.debug(f"RAG pipeline returned response for user {user_email}")

    except HTTPException as http_exc:
        raise http_exc # Re-raise FastAPI specific errors
    except Exception as e:
        logger.error(f"Unexpected error during RAG pipeline query for user {user_email} '{query_text[:100]}...': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during RAG processing.")

    # --- Append Interaction to History IF a valid conversation was loaded ---
    if conversation_id and result.answer and loaded_conv_data:
        logger.debug(f"Attempting to append interaction to conversation {conversation_id} for user {user_email}")
        try:
            current_time = datetime.now(timezone.utc)
            # Create new message objects - use ORIGINAL query text for history
            user_msg = ChatMessage(sender="user", text=query_text, timestamp=current_time)
            # Use getattr for safety in case 'sources' is missing from RAGResponse model instance
            ai_msg = ChatMessage(sender="ai", text=result.answer, sources=getattr(result, 'sources', None), timestamp=current_time)
            # Append to the loaded data's messages
            loaded_conv_data.messages.extend([user_msg, ai_msg])
            # Save the updated conversation data
            if memory_manager.save_conversation(user_email, loaded_conv_data):
                logger.info(f"Successfully appended interaction to conversation {conversation_id} for user {user_email}")
            else:
                # Log error if saving failed, but still return the RAG response
                logger.error(f"Failed to save updated history for conversation {conversation_id} after query (MemoryManager returned False).")
        except Exception as hist_e:
            # Log error if appending/saving history fails, but still return the RAG response
            logger.error(f"Error appending interaction to history for conversation {conversation_id}: {hist_e}", exc_info=True)
    elif conversation_id and not result.answer:
        # Log if no answer was generated, history won't be updated
        logger.warning(f"RAG pipeline did not return an answer for query in conversation {conversation_id}. History not updated.")
    # No action needed if conversation_id was None initially or became None because it wasn't found

    logger.info(f"Sending RAG response to user {user_email} for query: '{query_text[:100]}...'")
    return result


# --- MODIFIED Ingestion Endpoint (Requires Authentication) ---
# This endpoint handles PERMANENT ingestion, separate from temporary uploads
@app.post("/ingest_file", response_model=IngestResponse, tags=["Indexing"], status_code=status.HTTP_202_ACCEPTED, summary="Ingest and Index File Permanently")
async def ingest_file_endpoint(
    # Non-default background_tasks first
    background_tasks: BackgroundTasks,
    # Dependencies next
    current_user: Annotated[AuthUserData, Depends(get_current_user)],
    indexer: Annotated[Indexer, Depends(get_indexer)],
    summarizer: Annotated[DataSummarizer, Depends(get_summarizer)],
    # File parameter with default last
    file: UploadFile = File(...)
):
    """
    Schedules a file for background ingestion, summarization, and PERMANENT indexing.
    Uses a separate temporary location defined internally for this process.
    """
    user_email = current_user.email
    original_filename = file.filename if file.filename else "unknown_file"
    log_prefix = f"Permanent ingest request for user {user_email}: {original_filename}"
    logger.info(log_prefix)
    temp_path_str: Optional[str] = None

    try:
        # Define upload directory (consider making this configurable via settings)
        # This directory is specifically for the permanent ingestion process
        upload_dir = Path(settings.user_history_base_dir / "../uploads_ingest_temp") # Example: Separate temp dir
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Create a safe temporary filename
        safe_original_filename = clean_filename(original_filename)
        unique_suffix = str(uuid.uuid4())[:8] # Shorter suffix
        temp_filename = f"{unique_suffix}_{safe_original_filename}"
        # Basic length check (consider filesystem limits if necessary)
        max_len = 200
        if len(temp_filename) > max_len:
            name, ext = os.path.splitext(temp_filename)
            temp_filename = name[:max_len - len(ext) - 1] + '_' + ext if ext else name[:max_len-1] + '_'

        temp_file_path = upload_dir / temp_filename
        temp_path_str = str(temp_file_path)

        # Save the uploaded file temporarily for this specific process
        try:
            file_content = await file.read()
            with open(temp_file_path, "wb") as f:
                f.write(file_content)
            logger.info(f"Saved temporary file for permanent ingestion: {temp_path_str}")
        except Exception as save_err:
            logger.error(f"Failed to save temporary file {temp_path_str} for permanent ingestion: {save_err}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save uploaded file.")
        finally:
             # Ensure the uploaded file stream is closed
             try:
                 await file.close()
             except Exception as close_err:
                  logger.warning(f"Error closing file stream for permanent ingestion: {close_err}", exc_info=True)


        # Add the ingestion task to run in the background
        background_tasks.add_task(
            run_file_ingestion, # The background function defined earlier
            temp_path=temp_path_str, # Pass the path to the saved temp file
            filename=original_filename, # Pass the original filename for context
            indexer=indexer,
            summarizer=summarizer,
            user_email=user_email
        )

        logger.info(f"Scheduled background task for permanent ingestion: {original_filename} by user {user_email}")
        # Return success response immediately
        return IngestResponse(
            status="scheduled",
            message=f"Permanent ingestion task for '{original_filename}' scheduled successfully.",
            filename=original_filename
        )
    except HTTPException as http_exc:
        # If saving file failed, ensure temp file is cleaned up if it exists
        if temp_path_str and os.path.exists(temp_path_str):
            try: os.remove(temp_path_str)
            except OSError: logger.error(f"Failed to clean up temp file on HTTPException during permanent ingest schedule: {temp_path_str}")
        raise http_exc
    except Exception as e:
        logger.error(f"Failed to schedule permanent ingestion for {original_filename}: {e}", exc_info=True)
        # Clean up temp file on unexpected error during scheduling
        if temp_path_str and os.path.exists(temp_path_str):
            try: os.remove(temp_path_str)
            except OSError: logger.error(f"Failed to clean up temp file on error during permanent ingest schedule: {temp_path_str}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to schedule file ingestion due to a server error.")
