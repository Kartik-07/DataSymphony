# Standard library imports
import logging
import sys
import os
import uuid
import shutil
import schedule
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

# Third-party imports
import psycopg2
from psycopg2 import pool
from fastapi import UploadFile, HTTPException

# Optional imports for file reading (handle ImportError)
import pandas as pd
import pdfplumber
import docx2txt
import openpyxl # Needed by pandas for xlsx


# --- Existing setup_logging function ---
def setup_logging(level=logging.INFO):
    """Sets up basic logging configuration."""
    # Prevent adding handlers multiple times if called repeatedly
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        # Optionally clear existing handlers if re-configuration is desired
        # root_logger.handlers.clear()
        # Or just return if already configured
        return

    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    root_logger.setLevel(level)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)

    # Optional: File handler
    try:
        file_handler = logging.FileHandler("rag_system.log", encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        logging.error(f"Failed to configure file logging: {e}", exc_info=True)


    logging.info("Logging configured.")

# Note: It's generally better to call setup_logging once during application startup,
# for example, in your main.py or app factory.
# setup_logging()


# --- Optional: Postgres Pool (only needed for DIRECT SQL outside PGVector/SQLAlchemy) ---
class PostgresConnectionPool:
    """Minimal wrapper around psycopg2 pool for direct connections if needed."""
    _pool = None

    @classmethod
    def initialize(cls, dsn: str, minconn: int = 1, maxconn: int = 5):
        """Initializes the connection pool."""
        if cls._pool is None:
            try:
                cls._pool = pool.SimpleConnectionPool(minconn, maxconn, dsn=dsn)
                logging.info(f"Initialized PostgreSQL connection pool (min={minconn}, max={maxconn}).")
            except Exception as e:
                logging.error(f"Failed to create PostgreSQL connection pool: {e}", exc_info=True)
                cls._pool = None # Ensure it's None on failure
        else:
             logging.warning("PostgreSQL connection pool already initialized.")
        return cls._pool is not None

    @classmethod
    def getconn(cls):
        """Gets a connection from the pool."""
        if cls._pool:
            try:
                return cls._pool.getconn()
            except Exception as e:
                 logging.error(f"Failed to get connection from pool: {e}", exc_info=True)
                 raise ConnectionError("Failed to get connection from pool.") from e
        logging.error("Attempted to get connection from uninitialized pool.")
        raise ConnectionError("Postgres connection pool is not initialized.")

    @classmethod
    def putconn(cls, conn):
        """Returns a connection to the pool."""
        if cls._pool and conn:
             try:
                cls._pool.putconn(conn)
             except Exception as e:
                 logging.error(f"Failed to return connection to pool: {e}", exc_info=True)
                 # Depending on the error, you might want to close the connection instead
                 # conn.close()

    @classmethod
    def closeall(cls):
        """Closes all connections in the pool."""
        if cls._pool:
             cls._pool.closeall()
             logging.info("Closed PostgreSQL connection pool.")
             cls._pool = None

# --- Helper to get DSN for psycopg2 from SQLAlchemy URL ---
def get_psycopg2_dsn(sqlalchemy_url: str) -> str:
    """Converts a SQLAlchemy URL (like postgresql://user:pass@host:port/db) to a psycopg2 DSN string."""
    try:
        parsed = urlparse(sqlalchemy_url)
        # Ensure required components are present
        if not all([parsed.scheme.startswith('postgres'), parsed.username, parsed.password, parsed.hostname, parsed.port, parsed.path]):
             raise ValueError("Invalid SQLAlchemy URL format for DSN conversion.")

        dsn = f"dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} host={parsed.hostname} port={parsed.port}"
        return dsn
    except Exception as e:
        logging.error(f"Error converting SQLAlchemy URL to DSN: {e}", exc_info=True)
        raise ValueError("Could not parse SQLAlchemy URL for DSN.") from e


# --- Temporary File Upload Handling ---

# Configuration
# Determine project root relative to this file (utils.py)
# Assumes utils.py is in Backend directory, and temp_uploads should be at project root (one level up from Backend)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMP_UPLOAD_DIR = PROJECT_ROOT / "temp_uploads"
ALLOWED_EXTENSIONS = {".txt", ".docx", ".pdf", ".csv", ".xlsx"}
TEMP_FILE_LIFESPAN_HOURS = 24

# Ensure the temp directory exists on module load
try:
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Temporary upload directory ensured at: {TEMP_UPLOAD_DIR}")
except Exception as e:
    logging.error(f"Failed to create temporary upload directory {TEMP_UPLOAD_DIR}: {e}", exc_info=True)
    # Depending on severity, you might want to raise an error here to halt startup


def save_temporary_file(file: UploadFile) -> str:
    """
    Saves an uploaded file temporarily with a unique name and returns its unique ID (filename).
    Validates file extension and handles potential saving errors.
    """
    if not file.filename:
        logging.warning("Upload attempt with empty filename.")
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    _, extension = os.path.splitext(file.filename)
    if extension.lower() not in ALLOWED_EXTENSIONS:
        logging.warning(f"Upload attempt with disallowed file type: {extension}")
        raise HTTPException(status_code=400, detail=f"File type {extension} not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # Sanitize original filename stem for inclusion in the unique ID (improves readability)
    safe_original_stem = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in Path(file.filename).stem)
    unique_filename = f"{uuid.uuid4()}_{safe_original_stem}{extension}"
    file_path = TEMP_UPLOAD_DIR / unique_filename

    try:
        logging.info(f"Attempting to save temporary file: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Successfully saved temporary file: {unique_filename}")
    except Exception as e:
        logging.error(f"Could not save temporary file {unique_filename}: {e}", exc_info=True)
        # Clean up partial file if saving failed
        if file_path.exists():
            try:
                file_path.unlink()
            except OSError as unlink_err:
                 logging.error(f"Failed to clean up partial file {file_path}: {unlink_err}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        # Ensure the uploaded file stream is closed
        try:
            file.file.close()
        except Exception as close_err:
             logging.warning(f"Error closing uploaded file stream: {close_err}", exc_info=True)


    return unique_filename # Return only the unique filename (acts as ID within the temp dir)


def get_temporary_file_path(file_id: str) -> Path | None:
    """
    Gets the full, validated path of a temporary file if it exists and is within the designated temp directory.
    Performs basic security checks against path traversal.
    """
    if not file_id or not isinstance(file_id, str):
        logging.warning(f"Invalid file_id type or empty: {file_id}")
        return None

    # Basic check against directory traversal characters
    if '..' in file_id or '/' in file_id or '\\' in file_id:
        logging.warning(f"Potential path traversal attempt detected in file_id: {file_id}")
        return None

    try:
        # Construct the potential path
        file_path = (TEMP_UPLOAD_DIR / file_id).resolve()

        # Crucial Security Check: Ensure the resolved path is *still* within the intended TEMP_UPLOAD_DIR
        # This prevents resolving symlinks or complex paths outside the target directory.
        if TEMP_UPLOAD_DIR.resolve() in file_path.parents and file_path.exists() and file_path.is_file():
             return file_path
        else:
            logging.warning(f"Temporary file path validation failed or file not found for ID: {file_id}. Resolved path: {file_path}")
            return None
    except Exception as e:
        # Handle potential errors during path resolution or stat calls
        logging.error(f"Error resolving or checking temporary file path for ID {file_id}: {e}", exc_info=True)
        return None


def read_temporary_file_content(file_id: str) -> str:
    """
    Reads the content of a temporary file based on its extension, using appropriate libraries.
    Handles potential errors during file reading and library usage.
    """
    file_path = get_temporary_file_path(file_id)
    if not file_path:
        logging.error(f"Attempted to read non-existent or invalid temporary file with ID: {file_id}")
        raise HTTPException(status_code=404, detail=f"Temporary file '{file_id}' not found or invalid.")

    extension = file_path.suffix.lower()
    content = ""
    logging.info(f"Reading temporary file: {file_path} (Extension: {extension})")

    try:
        if extension == ".txt":
            # Read plain text file
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                content = f.read()
        elif extension == ".csv":
            # Read CSV file using pandas if available
            if pd:
                try:
                    df = pd.read_csv(file_path)
                    # Convert entire DataFrame to string for LLM context (adjust if needed)
                    content = df.to_string()
                except Exception as e:
                    logging.error(f"Pandas failed to read CSV {file_id}: {e}", exc_info=True)
                    content = f"[Error reading CSV with pandas: {e}]"
            else:
                logging.warning("Pandas library not installed, attempting basic text read for CSV.")
                # Fallback to simple text reading if pandas isn't available
                with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                    content = "[Reading CSV as plain text - pandas not installed]\n" + f.read()

        elif extension == ".pdf":
            # Read PDF using pdfplumber if available
            if pdfplumber:
                try:
                    text_parts = []
                    with pdfplumber.open(file_path) as pdf:
                        if not pdf.pages:
                             logging.warning(f"PDF file {file_id} has no pages.")
                             content = "[PDF file has no pages]"
                        else:
                            for i, page in enumerate(pdf.pages):
                                page_text = page.extract_text()
                                if page_text: # Check if text was extracted
                                    text_parts.append(page_text)
                                else:
                                     logging.warning(f"No text extracted from page {i+1} of PDF {file_id}.")
                    content = "\n\n".join(text_parts) # Join pages with double newline
                    if not content:
                        content = "[No text could be extracted from PDF]"
                except Exception as e:
                    logging.error(f"pdfplumber failed to read PDF {file_id}: {e}", exc_info=True)
                    content = f"[Error reading PDF with pdfplumber: {e}]"
            else:
                logging.error("pdfplumber library not installed, cannot read PDF content.")
                content = "[Could not read PDF: pdfplumber library not installed]"

        elif extension == ".docx":
            # Read DOCX using docx2txt if available
            if docx2txt:
                try:
                    content = docx2txt.process(file_path)
                except Exception as e:
                    logging.error(f"docx2txt failed to read DOCX {file_id}: {e}", exc_info=True)
                    content = f"[Error reading DOCX with docx2txt: {e}]"
            else:
                logging.error("docx2txt library not installed, cannot read DOCX content.")
                content = "[Could not read DOCX: docx2txt library not installed]"

        elif extension == ".xlsx":
            # Read XLSX using pandas if available (requires openpyxl)
            if pd and openpyxl:
                try:
                    xls = pd.ExcelFile(file_path)
                    sheet_contents = []
                    if not xls.sheet_names:
                         logging.warning(f"XLSX file {file_id} contains no sheets.")
                         content = "[XLSX file contains no sheets]"
                    else:
                        for sheet_name in xls.sheet_names:
                            df = pd.read_excel(xls, sheet_name=sheet_name)
                            # Convert sheet DataFrame to string
                            sheet_contents.append(f"--- Sheet: {sheet_name} ---\n{df.to_string()}")
                        content = "\n\n".join(sheet_contents) # Join sheets with double newline
                except Exception as e:
                    logging.error(f"Pandas failed to read XLSX {file_id}: {e}", exc_info=True)
                    content = f"[Error reading XLSX with pandas: {e}]"
            else:
                 missing_lib = ""
                 if not pd: missing_lib += "pandas "
                 if not openpyxl: missing_lib += "openpyxl"
                 logging.error(f"{missing_lib.strip()} library not installed, cannot read XLSX content.")
                 content = f"[Could not read XLSX: {missing_lib.strip()} library not installed]"
        else:
             # Should not happen if ALLOWED_EXTENSIONS is enforced, but good practice
             logging.warning(f"Attempted to read file with unsupported extension '{extension}' for ID: {file_id}")
             content = "[Unsupported file type for content extraction]"

    except Exception as e:
        # Catch-all for unexpected errors during file processing
        logging.error(f"Unexpected error processing file {file_id}: {e}", exc_info=True)
        content = f"[Unexpected error processing file {file_id}]"

    # Optional: Add content truncation logging or logic here if needed
    # MAX_LOG_LEN = 100
    # logging.debug(f"Read content for {file_id} (first {MAX_LOG_LEN} chars): {content[:MAX_LOG_LEN]}")

    return content


# --- Cleanup Scheduler for Temporary Files ---

def delete_old_temp_files():
    """Scans the temporary directory and deletes files older than the configured lifespan."""
    now = datetime.now(timezone.utc) # Use timezone-aware datetime
    cutoff = now - timedelta(hours=TEMP_FILE_LIFESPAN_HOURS)
    logging.info(f"Running cleanup for temporary files older than {cutoff} in {TEMP_UPLOAD_DIR}...")
    deleted_count = 0
    error_count = 0

    try:
        for item in TEMP_UPLOAD_DIR.iterdir():
            if item.is_file(): # Ensure it's a file
                try:
                    # Get modification time and make it timezone-aware (UTC)
                    mod_time_timestamp = item.stat().st_mtime
                    mod_time = datetime.fromtimestamp(mod_time_timestamp, timezone.utc)

                    if mod_time < cutoff:
                        item.unlink() # Delete the file
                        logging.info(f"Deleted old temporary file: {item.name}")
                        deleted_count += 1
                except OSError as e:
                    logging.error(f"Error deleting file {item.name}: {e}", exc_info=True)
                    error_count += 1
                except Exception as e:
                     logging.error(f"Error processing file {item.name} during cleanup: {e}", exc_info=True)
                     error_count += 1
        logging.info(f"Temporary file cleanup finished. Deleted: {deleted_count}, Errors: {error_count}.")
    except Exception as e:
        logging.error(f"Error during temporary file cleanup scan of directory {TEMP_UPLOAD_DIR}: {e}", exc_info=True)


def run_scheduler():
    """Runs the cleanup schedule in a loop. Designed to be run in a background thread."""
    logging.info("Background scheduler thread started.")
    # Schedule the job every hour (adjust frequency as needed)
    schedule.every(1).hour.do(delete_old_temp_files)

    # Run once immediately on startup of this thread
    try:
        delete_old_temp_files()
    except Exception as e:
        logging.error(f"Initial run of delete_old_temp_files failed: {e}", exc_info=True)


    while True:
        try:
            schedule.run_pending()
        except Exception as e:
             # Log errors during schedule execution but keep the loop running
             logging.error(f"Error during schedule.run_pending(): {e}", exc_info=True)
        # Sleep for a reasonable interval (e.g., 60 seconds)
        time.sleep(60)


# --- Function to Start Scheduler in Background ---
# This should be called once when the main application starts (e.g., in main.py startup event)
_scheduler_thread = None
def start_cleanup_scheduler():
    """Starts the background cleanup scheduler thread if not already running."""
    global _scheduler_thread
    if _scheduler_thread is None or not _scheduler_thread.is_alive():
        logging.info("Starting background cleanup scheduler thread...")
        # Use daemon=True so the thread doesn't block application exit
        _scheduler_thread = threading.Thread(target=run_scheduler, daemon=True, name="TempFileCleanupScheduler")
        _scheduler_thread.start()
        logging.info("Background cleanup scheduler thread started.")
    else:
        logging.warning("Cleanup scheduler thread already running.")
