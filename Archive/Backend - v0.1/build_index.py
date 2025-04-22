import logging
import os
from config import settings
from utils import setup_logging
from data_processing import DataLoader, TextProcessor
from indexing import Indexer

setup_logging(level=logging.INFO) # Set level if needed
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Starting Indexing Process ---")
    target_folder = r"C:/Users/karti/OneDrive/Desktop/RAG/Document" # Use your actual path
    all_docs = []

    if not os.path.isdir(target_folder):
        logger.error(f"Target folder not found or is not a directory: {target_folder}")
        return

    logger.info(f"Scanning folder for documents: {target_folder}")
    for filename in os.listdir(target_folder):
        file_path = os.path.join(target_folder, filename)
        if os.path.isfile(file_path):
            logger.debug(f"Processing file: {filename}")
            docs = []
            file_ext = filename.lower().split('.')[-1] # Get extension

            try:
                if file_ext == "pdf":
                    logger.info(f"Loading PDF: {filename}")
                    docs = DataLoader.load_pdf(file_path)
                elif file_ext == "txt":
                    logger.info(f"Loading TXT: {filename}")
                    docs = DataLoader.load_text(file_path)
                # --- ADDED DOCX HANDLING ---
                elif file_ext == "docx":
                     logger.info(f"Loading DOCX: {filename}")
                     docs = DataLoader.load_docx(file_path)
                # --- END DOCX HANDLING ---
                else:
                    logger.warning(f"Skipping unsupported file type: {filename}")
                    continue

                if docs:
                    all_docs.extend(docs)
                    logger.info(f"Successfully loaded {len(docs)} document part(s) from: {filename}")
                else:
                     logger.warning(f"No documents loaded from file (might be empty or issue with loader): {filename}")

            except Exception as e:
                logger.error(f"Failed to load or process file {filename}: {e}", exc_info=True)
        else:
            logger.debug(f"Skipping item (it's a directory or other non-file type): {filename}")

    if not all_docs:
        logger.error("No documents were successfully loaded from the specified folder. Aborting indexing.")
        return

    logger.info(f"Loaded a total of {len(all_docs)} document parts from the folder.")

    processor = TextProcessor()
    splits = processor.split_documents(all_docs)
    logger.info(f"Number of chunks created: {len(splits)}")

    if not splits:
        logger.error("No chunks were created after splitting. Aborting indexing.")
        return

    try:
        indexer = Indexer()
        logger.info(f"Passing {len(splits)} chunks to indexer...")
        indexer.index_documents(splits)
        logger.info("--- Indexing Process Completed Successfully ---")
    except Exception as e:
        logger.critical(f"--- Indexing Process Failed: {e} ---", exc_info=True)

if __name__ == "__main__":
    main()