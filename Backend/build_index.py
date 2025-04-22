# Backend/build_index.py

import logging
import os
import json
import re # <-- Import re for table name cleaning
from typing import List, Dict, Any, Optional

# Import necessary components using absolute paths
from config import settings
from utils import setup_logging
from data_processing import DataLoader, TextProcessor
from indexing import Indexer, ensure_pgvector_setup
from summarization import DataSummarizer
from sql_processing import get_rag_db_utility, SQLDatabase
from langchain_core.documents import Document
import pandas as pd
from sqlalchemy import create_engine # <-- Import create_engine

# --- Logging Setup ---
setup_logging(level=logging.INFO) # Set desired log level
logger = logging.getLogger(__name__)

# --- Helper function to clean filename for SQL table name ---
def clean_table_name(filename: str) -> str:
    """Cleans a filename to create a valid SQL table name."""
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]
    # Replace non-alphanumeric characters (except underscore) with underscore
    cleaned_name = re.sub(r'[^\w]+', '_', name_without_ext)
    # Ensure it starts with a letter or underscore
    if not re.match(r'^[a-zA-Z_]', cleaned_name):
        cleaned_name = '_' + cleaned_name
    # Convert to lowercase
    cleaned_name = cleaned_name.lower()
    # Truncate if too long (optional, depends on DB limits)
    max_len = 63 # PostgreSQL default identifier limit
    return cleaned_name[:max_len]

# --- Main Indexing Function ---
def main():
    logger.info("--- Starting Data Loading, Metadata Generation, Chunking, and Indexing Process ---")

    # --- Database Pre-check (for RAG_DB / PGVector) ---
    try:
        logger.info("Ensuring database (PGVector for RAG_DB) is set up...")
        ensure_pgvector_setup() # This targets settings.postgres_url (RAG_DB)
        logger.info("RAG_DB setup check complete.")
    except Exception as db_err:
         logger.critical(f"RAG_DB setup failed: {db_err}. Aborting.", exc_info=True)
         return

    # --- Create Engine for Uploads DB ---
    uploads_engine = None
    try:
        logger.info(f"Creating SQLAlchemy engine for Uploads DB: {settings.postgres_uploads_url}")
        uploads_engine = create_engine(settings.postgres_uploads_url)
        # Optionally, test connection
        with uploads_engine.connect() as connection:
            logger.info("Successfully connected to Uploads DB (RAG_DB_UPLOADS).")
    except Exception as engine_err:
        logger.error(f"Failed to create engine or connect to Uploads DB: {engine_err}. CSV/XLSX files will not be loaded into DB.", exc_info=True)
        # Allow script to continue to process other files/metadata, but uploads won't work.

    # --- Initialize Components ---
    db_utility: Optional[SQLDatabase] = None # For RAG_DB SQL queries
    summarizer: Optional[DataSummarizer] = None
    indexer: Optional[Indexer] = None # For RAG_DB Vector Store
    processor: Optional[TextProcessor] = None

    try:
         # Initialize DB Utility for RAG_DB (used by summarizer for table schema)
         logger.info("Initializing DB Utility for RAG_DB...")
         db_utility = get_rag_db_utility() # Connects to settings.postgres_url
         if not db_utility:
              logger.critical("Failed to initialize DB Utility for RAG_DB. Table processing will be skipped.")
         else:
              logger.info("RAG_DB utility obtained successfully.")

         logger.info("Initializing Indexer (for RAG_DB)...")
         indexer = Indexer() # Connects to settings.postgres_url
         logger.info("Indexer initialized.")

         logger.info("Initializing DataSummarizer (for metadata)...")
         summarizer = DataSummarizer()
         # Check internal DB connection of summarizer (should be RAG_DB)
         if not summarizer.db_utility and db_utility:
             logger.warning("Summarizer failed to get RAG_DB utility internally.")
         elif not summarizer.db_utility and not db_utility:
              logger.warning("Summarizer confirms RAG_DB utility is unavailable.")
         logger.info(f"DataSummarizer initialized with model '{settings.light_llm_model_name}'.")

         logger.info("Initializing TextProcessor...")
         processor = TextProcessor(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
         logger.info("TextProcessor initialized.")

    except Exception as init_err:
         logger.critical(f"Failed to initialize components: {init_err}. Aborting.", exc_info=True)
         return

    # Ensure essential components initialized
    if not indexer or not summarizer or not processor:
         logger.critical("Essential components (Indexer, Summarizer, Processor) failed to initialize. Aborting.")
         return

    # --- Configuration for Target Data ---
    target_folder = os.getenv("INDEXING_TARGET_FOLDER", r"C:/Users/karti/OneDrive/Desktop/RAG/Document")
    logger.info(f"Target document folder: {target_folder}")

    if not os.path.isdir(target_folder):
        logger.error(f"Target folder not found: {target_folder}")
        return

    all_final_chunks: List[Document] = [] # Chunks from TXT, PDF, DOCX for RAG_DB index
    metadata_docs_to_index: List[Document] = [] # Metadata docs (CSV, XLSX, DB Tables) for RAG_DB index
    all_metadata_list: List[Dict[str, Any]] = [] # List to save metadata.json (includes errors)

    processed_files = 0; skipped_files = 0; unsupported_files = 0; db_uploads = 0; db_upload_failures = 0

    # --- File Processing Loop ---
    logger.info(f"Scanning folder for documents: {target_folder}")
    for filename in os.listdir(target_folder):
        file_path = os.path.join(target_folder, filename)
        if not os.path.isfile(file_path) or filename.startswith('.') or filename.lower() == "metadata.json":
            logger.debug(f"Skipping item: {filename}"); continue

        logger.info(f"--- Processing file: {filename} ---")
        file_ext = filename.lower().split('.')[-1]
        file_metadata: Optional[Dict[str, Any]] = None
        doc_summary = "Summary generation failed or skipped." # Default for unstructured

        try:
            # --- Handle Unstructured Files (TXT, PDF, DOCX) ---
            if file_ext in ["pdf", "txt", "docx"]:
                # ... (Existing logic for loading, summarizing, chunking, and adding to all_final_chunks) ...
                # --- (Code identical to previous version - omitted for brevity) ---
                logger.info(f"Processing unstructured file type: .{file_ext}")
                loaded_docs: List[Document] = []
                if file_ext == "pdf": loaded_docs = DataLoader.load_pdf(file_path)
                elif file_ext == "txt": loaded_docs = DataLoader.load_text(file_path)
                elif file_ext == "docx": loaded_docs = DataLoader.load_docx(file_path)

                if not loaded_docs:
                     logger.warning(f"No content loaded from: {filename}"); skipped_files += 1
                     file_metadata = {"id": f"error-{filename}-load_failed", "document": "Load failure or empty file.", "metadata": {"identifier": filename, "data_type": "error", "error": "Load failure or empty file", "source_type": "file"}}
                     all_metadata_list.append(file_metadata); continue

                logger.info(f"Loaded {len(loaded_docs)} parts from: {filename}"); processed_files += 1
                full_doc_content = "\n\n".join([doc.page_content for doc in loaded_docs if doc.page_content])
                if not full_doc_content:
                    logger.warning(f"No text content extracted for: {filename}"); skipped_files +=1; processed_files -=1
                    file_metadata = {"id": f"error-{filename}-no_content", "document": "No text content extracted.", "metadata": {"identifier": filename, "data_type": "error", "error": "No text content extracted", "source_type": "file"}}
                    all_metadata_list.append(file_metadata); continue

                # Generate Summary & Full Metadata for unstructured
                try:
                     file_metadata = summarizer.summarize(preloaded_content=full_doc_content, summary_method='unstructured', file_name_override=filename)
                     if file_metadata and file_metadata.get("metadata", {}).get("data_type") != "error":
                          doc_summary = file_metadata.get("document", "Summary could not be extracted.")
                          logger.info(f"Generated summary and metadata for unstructured: {filename}")
                          all_metadata_list.append(file_metadata)
                     else:
                          err_msg = file_metadata.get('metadata', {}).get('error', 'Unknown summarization error') if file_metadata else 'Summarizer returned None'
                          logger.warning(f"Failed to generate summary/metadata for {filename}. Error: {err_msg}")
                          error_meta = {"id": f"error-{filename}-summarization_failed", "document": f"Summarization failed: {err_msg}", "metadata": {"identifier": filename, "data_type": "error", "error": f"Summarization failed: {err_msg}", "source_type": "file"}}
                          all_metadata_list.append(error_meta)
                          doc_summary = f"Summary generation failed: {err_msg}"
                except Exception as summ_err:
                     logger.error(f"Error during summary generation for {filename}: {summ_err}", exc_info=True)
                     error_meta = {"id": f"error-{filename}-summarization_exception", "document": f"Summarization exception: {summ_err}", "metadata": {"identifier": filename, "data_type": "error", "error": f"Summarization exception: {summ_err}", "source_type": "file"}}
                     all_metadata_list.append(error_meta)
                     doc_summary = f"Summary generation exception: {summ_err}"

                # Split Unstructured Document into Chunks
                logger.info(f"Splitting content from {filename} into chunks...")
                chunks: List[Document] = processor.split_documents(loaded_docs)
                logger.info(f"Created {len(chunks)} chunks for: {filename}")
                if not chunks: logger.warning(f"No chunks created for file {filename}."); continue

                # Add Metadata to Chunks and Collect for Indexing in RAG_DB
                for chunk_index, chunk in enumerate(chunks):
                    if not hasattr(chunk, 'metadata') or chunk.metadata is None: chunk.metadata = {}
                    chunk.metadata['original_doc_summary'] = doc_summary # Add summary to chunk metadata
                    if 'source' not in chunk.metadata: chunk.metadata['source'] = filename
                    chunk.metadata['original_file_path'] = file_path
                    chunk.metadata['chunk_index_in_doc'] = chunk_index
                    all_final_chunks.append(chunk) # Add chunk to list for indexing

            # --- Handle Structured Files (CSV, XLSX) ---
            elif file_ext in ["csv", "xlsx"]:
                logger.info(f"Processing structured file type: .{file_ext}")
                df: Optional[pd.DataFrame] = None
                try:
                    if file_ext == "csv":
                        df = pd.read_csv(file_path)
                    else: # xlsx
                        # Load the first sheet by default, requires openpyxl
                        df = pd.read_excel(file_path, sheet_name=0)
                    logger.info(f"Loaded DataFrame from {filename}. Shape: {df.shape if df is not None else 'None'}")
                    processed_files += 1
                except ImportError as ie:
                     # Specifically handle missing dependency for Excel
                     logger.error(f"Failed to load {filename}: Missing dependency {ie}. Install it (e.g., pip install openpyxl)")
                     skipped_files += 1; processed_files -=1 if processed_files > 0 else 0
                     file_metadata = {"id": f"error-{filename}-missing_dep", "document": f"Load failed: Missing dependency {ie}", "metadata": {"identifier": filename, "data_type": "error", "source_type": "file", "error": f"Load failed: Missing dependency {ie}"}}
                     all_metadata_list.append(file_metadata); continue
                except Exception as load_err:
                     logger.error(f"Failed to load structured file {filename}: {load_err}", exc_info=True)
                     skipped_files += 1; processed_files -=1 if processed_files > 0 else 0
                     file_metadata = {"id": f"error-{filename}-struct_load_failed", "document": f"Failed to load: {load_err}", "metadata": {"identifier": filename, "data_type": "error", "source_type": "file", "error": f"Failed to load structured file: {load_err}"}}
                     all_metadata_list.append(file_metadata); continue

                if df is None or df.empty:
                    logger.warning(f"Structured file is empty or loading failed: {filename}")
                    # Don't count as skipped if loading succeeded but was empty
                    if df is None: skipped_files +=1; processed_files -=1 if processed_files > 0 else 0
                    # Create basic metadata indicating empty file
                    file_metadata = {"id": f"info-{filename}-empty_structured", "document": "Empty structured file (CSV/XLSX).", "metadata": {"identifier": filename, "data_type": "structured", "source_type": "file", "row_count": 0, "column_count": 0, "columns": [], "enrichment_status": "not_applicable"}}
                    all_metadata_list.append(file_metadata)
                    # Don't try to write empty df to DB or create metadata doc for index
                    continue

                # --- Write DataFrame to RAG_DB_UPLOADS ---
                table_name_cleaned = clean_table_name(filename)
                if uploads_engine: # Only proceed if engine was created successfully
                    try:
                        logger.info(f"Attempting to write DataFrame from '{filename}' to table '{table_name_cleaned}' in RAG_DB_UPLOADS...")
                        # Use 'replace' to overwrite table if it exists during re-indexing
                        df.to_sql(name=table_name_cleaned, con=uploads_engine, if_exists='replace', index=False, chunksize=1000) # Added chunksize for larger files
                        logger.info(f"Successfully wrote '{filename}' to DB table '{table_name_cleaned}'.")
                        db_uploads += 1
                    except Exception as db_write_err:
                        logger.error(f"Failed to write DataFrame from '{filename}' to DB table '{table_name_cleaned}': {db_write_err}", exc_info=True)
                        db_upload_failures += 1
                        # Add error info specifically about DB write failure
                        error_meta_db = {"id": f"error-{filename}-db_write_failed", "document": f"DB write failed: {db_write_err}", "metadata": {"identifier": filename, "target_table": table_name_cleaned, "data_type": "structured", "source_type": "file", "error": f"DB write failed: {db_write_err}"}}
                        all_metadata_list.append(error_meta_db)
                        # Continue to generate metadata for indexing even if DB write fails
                else:
                    logger.warning(f"Skipping DB upload for '{filename}' as Uploads DB engine is not available.")
                    db_upload_failures += 1
                # --- End Write DataFrame ---

                # --- Generate Summary & Metadata for Indexing in RAG_DB ---
                try:
                     # Use 'llm' method for enrichment by default for structured files
                     file_metadata = summarizer.summarize(data_input=df, summary_method='llm', file_name_override=filename)
                     if file_metadata and file_metadata.get("metadata", {}).get("data_type") != "error":
                          logger.info(f"Generated metadata for structured file: {filename}")
                          all_metadata_list.append(file_metadata) # Add to metadata.json list

                          # Add info about the target DB table to the metadata before indexing
                          file_metadata["metadata"]["target_database"] = "RAG_DB_UPLOADS"
                          file_metadata["metadata"]["target_table_name"] = table_name_cleaned

                          # Create single Document to index in RAG_DB Vector Store
                          metadata_doc = Document(
                              page_content=file_metadata.get("document", f"Summary for {filename}"), # Use generated description/summary
                              metadata=file_metadata.get("metadata", {"identifier": filename, "error": "Metadata missing"}) # Embed full metadata
                          )
                          metadata_docs_to_index.append(metadata_doc)
                     else:
                          err_msg = file_metadata.get('metadata', {}).get('error', 'Unknown summarization error') if file_metadata else 'Summarizer returned None'
                          logger.warning(f"Failed to generate metadata for structured file {filename}. Error: {err_msg}")
                          error_meta_summ = {"id": f"error-{filename}-struct_summarization_failed", "document": f"Summarization failed: {err_msg}", "metadata": {"identifier": filename, "data_type": "error", "source_type": "file", "error": f"Summarization failed: {err_msg}"}}
                          all_metadata_list.append(error_meta_summ)
                except Exception as summ_err:
                     logger.error(f"Error during structured summary generation for {filename}: {summ_err}", exc_info=True)
                     error_meta_summ_exc = {"id": f"error-{filename}-struct_summarization_exception", "document": f"Summarization exception: {summ_err}", "metadata": {"identifier": filename, "data_type": "error", "source_type": "file", "error": f"Summarization exception: {summ_err}"}}
                     all_metadata_list.append(error_meta_summ_exc)
                # --- End Generate Summary ---

            # --- Handle Unsupported Files ---
            else:
                logger.warning(f"Skipping unsupported file type: {filename}"); unsupported_files += 1
                file_metadata = {"id": f"error-{filename}-unsupported", "document": f"Unsupported file type '.{file_ext}'.", "metadata": {"identifier": filename, "data_type": "error", "error": "Unsupported file type", "source_type": "file"}}
                all_metadata_list.append(file_metadata); continue

        # --- General Exception Handling for File Loop ---
        except ImportError as ie: # Catch import errors early in the loop
             logger.error(f"Skipping {filename}: Missing library required for processing: {ie}.", exc_info=False); skipped_files += 1; processed_files -= 1 if processed_files > 0 else 0
             error_meta = {"id": f"error-{filename}-import_error", "document": f"ImportError: {ie}", "metadata": {"identifier": filename, "data_type": "error", "error": f"ImportError: {ie}", "source_type": "file"}}
             all_metadata_list.append(error_meta)
        except Exception as e:
            logger.error(f"Failed processing {filename}: {e}", exc_info=True); skipped_files += 1; processed_files -= 1 if processed_files > 0 else 0
            error_meta = {"id": f"error-{filename}-processing_exception", "document": f"Processing exception: {e}", "metadata": {"identifier": filename, "data_type": "error", "error": f"Processing exception: {e}", "source_type": "file"}}
            all_metadata_list.append(error_meta)
    # --- End of File Processing Loop ---


    logger.info(f"--- File Processing Complete ---")
    logger.info(f"Successfully processed content/metadata for {processed_files} files.")
    if db_uploads > 0: logger.info(f"Successfully uploaded {db_uploads} CSV/XLSX files to RAG_DB_UPLOADS.")
    if unsupported_files > 0: logger.warning(f"Skipped {unsupported_files} unsupported file types.")
    if skipped_files > 0: logger.warning(f"Skipped {skipped_files} files due to loading errors.")
    if db_upload_failures > 0: logger.warning(f"Failed to upload {db_upload_failures} CSV/XLSX files to database.")
    logger.info(f"Prepared {len(all_final_chunks)} text chunks for indexing in RAG_DB.")
    logger.info(f"Prepared {len(metadata_docs_to_index)} metadata documents (CSV/XLSX/DB Tables) for indexing in RAG_DB.")

    # --- Database Table Metadata Generation (For RAG_DB only) ---
    if db_utility and summarizer.db_utility: # Only process RAG_DB tables
        logger.info("--- Processing RAG_DB Database Tables for Metadata ---")
        tables_to_exclude = {"langchain_pg_collection", "langchain_pg_embedding"}
        try:
            table_names = db_utility.get_usable_table_names()
            tables_to_process = [name for name in table_names if name not in tables_to_exclude]
            logger.info(f"Found {len(tables_to_process)} tables in RAG_DB to process: {tables_to_process}")

            for table_name in tables_to_process:
                logger.info(f"--- Generating metadata for RAG_DB table: {table_name} ---")
                try:
                    table_metadata = summarizer.summarize(table_name=table_name, summary_method='llm') # Uses RAG_DB connection inside summarizer
                    if table_metadata and table_metadata.get("metadata", {}).get("data_type") != "error":
                         all_metadata_list.append(table_metadata) # Add to metadata.json list
                         logger.info(f"Successfully generated metadata for RAG_DB table: {table_name}")

                         # Add info about the source DB table to the metadata before indexing
                         table_metadata["metadata"]["target_database"] = "RAG_DB" # Indicate source DB
                         table_metadata["metadata"]["target_table_name"] = table_name

                         # Create single Document to index in RAG_DB Vector Store
                         metadata_doc = Document(
                              page_content=table_metadata.get("document", f"Summary for RAG_DB table {table_name}"),
                              metadata=table_metadata.get("metadata", {"identifier": table_name, "error": "Metadata missing"})
                          )
                         metadata_docs_to_index.append(metadata_doc) # Add to indexing list
                    else:
                         err_msg = table_metadata.get('metadata', {}).get('error', 'Unknown table summarization error') if table_metadata else 'Summarizer returned None'
                         logger.error(f"Failed to generate metadata for RAG_DB table {table_name}. Error: {err_msg}")
                         if table_metadata: all_metadata_list.append(table_metadata)
                         else: error_meta = {"id": f"error-table-{table_name}-summarizer_returned_none", "document": "Summarizer returned None.", "metadata": {"identifier": table_name, "data_type": "error", "source_type": "database_table", "error": "Summarizer returned None"}}; all_metadata_list.append(error_meta)
                except Exception as table_err:
                    logger.error(f"Exception while summarizing RAG_DB table {table_name}: {table_err}", exc_info=True)
                    error_meta = {"id": f"error-table-{table_name}-summarization_exception", "document": f"Table summarization exception: {table_err}", "metadata": {"identifier": table_name, "data_type": "error", "source_type": "database_table", "error": f"Summarization exception: {table_err}"}}; all_metadata_list.append(error_meta)
        except Exception as db_list_err:
            logger.error(f"Failed to get table names from RAG_DB: {db_list_err}", exc_info=True)
            error_meta = {"id": f"error-db-list_tables_failed", "document": f"Failed list RAG_DB tables: {db_list_err}", "metadata": {"identifier": "RAG_DB", "data_type": "error", "source_type": "database", "error": f"Failed list tables: {db_list_err}"}}; all_metadata_list.append(error_meta)
    else:
        logger.warning("Skipping RAG_DB table metadata generation as DB utility is not available or summarizer could not access it.")
        error_meta = {"id": f"error-db-connection_unavailable", "document": "RAG_DB connection unavailable, skipping table metadata.", "metadata": {"identifier": "RAG_DB", "data_type": "error", "source_type": "database", "error": "RAG_DB connection unavailable for table processing."}}; all_metadata_list.append(error_meta)
    # logger.info(f"Prepared {len(metadata_docs_to_index) - len(all_final_chunks)} metadata documents (DB tables) for indexing.") # Logic might be off here


    # --- Write Combined Metadata to JSON File ---
    metadata_file_path = os.path.join(target_folder, "metadata.json")
    logger.info(f"--- Writing Combined Metadata to: {metadata_file_path} ---")
    try:
        os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
        with open(metadata_file_path, 'w', encoding='utf-8') as f: json.dump(all_metadata_list, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully wrote metadata for {len(all_metadata_list)} items.")
    except IOError as io_err: logger.error(f"Failed to write metadata file: {io_err}", exc_info=True)
    except TypeError as json_err: logger.error(f"Failed to serialize metadata to JSON: {json_err}.", exc_info=True)


# --- Final Indexing Step (Combine Chunks and Metadata Docs into RAG_DB Vector Store) ---
    combined_docs_for_indexing = all_final_chunks + metadata_docs_to_index
    if not combined_docs_for_indexing:
        logger.warning("No document chunks or metadata documents prepared for indexing in RAG_DB. Check logs."); logger.info("--- Indexing Process Skipped ---")
        return

    # --- ADD NUL BYTE CLEANING START ---
    logger.info("Cleaning NUL bytes from document content before indexing...")
    cleaned_count = 0
    nul_byte_sources = [] # Optional: Track sources with NUL bytes

    for i, doc in enumerate(combined_docs_for_indexing):
        cleaned_content = doc.page_content
        try:
            # Clean page_content (main document text)
            if isinstance(doc.page_content, str) and '\x00' in doc.page_content:
                cleaned_content = doc.page_content.replace('\x00', '')
                if doc.page_content != cleaned_content:
                    doc.page_content = cleaned_content
                    cleaned_count += 1
                    source_info = doc.metadata.get('source', f'doc_index_{i}')
                    if source_info not in nul_byte_sources:
                        nul_byte_sources.append(source_info)

            # Optional: Clean string values within metadata dictionary
            if isinstance(doc.metadata, dict):
                 for key, value in doc.metadata.items():
                     if isinstance(value, str) and '\x00' in value:
                          cleaned_value = value.replace('\x00', '')
                          if value != cleaned_value:
                              doc.metadata[key] = cleaned_value
                              # No need to increment cleaned_count again, just ensure data is clean
                              source_info = doc.metadata.get('source', f'doc_index_{i}_meta_{key}')
                              if source_info not in nul_byte_sources:
                                   nul_byte_sources.append(source_info)

        except Exception as clean_err:
             logger.error(f"Error cleaning document at index {i} (Source: {doc.metadata.get('source', 'N/A')}): {clean_err}", exc_info=False)
             # Decide if you want to skip this doc or try indexing anyway
             # Option: Remove the problematic doc: combined_docs_for_indexing.pop(i); continue # Be careful with list modification while iterating

    if cleaned_count > 0:
         logger.info(f"Removed NUL ('\\x00') bytes from content/metadata of {cleaned_count} documents.")
         if nul_byte_sources:
             logger.warning(f"NUL bytes were found in sources (or their metadata): {', '.join(nul_byte_sources[:10])}{'...' if len(nul_byte_sources) > 10 else ''}")
    else:
         logger.info("No NUL ('\\x00') bytes found in document content/metadata during pre-indexing check.")
    # --- ADD NUL BYTE CLEANING END ---


    logger.info(f"Attempting to index {len(combined_docs_for_indexing)} total documents (text chunks + metadata summaries) into RAG_DB...")
    try:
        # indexer connects to RAG_DB (settings.postgres_url)
        indexer.index_documents(combined_docs_for_indexing) # Now indexing cleaned documents
        logger.info(f"--- Indexing Process Completed Successfully for {len(combined_docs_for_indexing)} documents into RAG_DB ---")
    except Exception as e:
        logger.critical(f"--- Indexing Process Failed: {e} ---", exc_info=True) # This error should now be different if cleaning worked

# --- Script Execution ---
if __name__ == "__main__":
    main()