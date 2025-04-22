import logging
from typing import List
import bs4
from langchain_community.document_loaders import WebBaseLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader # Use PyPDF if preferred
from langchain_community.document_loaders import PDFPlumberLoader # Added from script 2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import settings
import os

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads data from various sources."""

    @staticmethod
    def load_web_page(url: str) -> List[Document]:
        """Loads content from a single web page."""
        # --- No changes needed here, keeping your implementation ---
        try:
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header", "content", "main") # Add common tags
                    )
                ),
            )
            loader.requests_per_second = 1 # Respectful scraping
            docs = loader.load()
            # Add URL as metadata if not present
            for doc in docs:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = url
            logger.info(f"Loaded content from {url}")
            return docs
        except Exception as e:
            logger.error(f"Error loading web page {url}: {e}", exc_info=True)
            return []

    # VVVVVV RENAMED this method VVVVVV
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]: # Renamed from load_pdf_plumber
        """Loads content from a PDF file using PDFPlumberLoader."""
        logger.info(f"Attempting to load PDF: {file_path} using PDFPlumber")
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            return []
        try:
            # Using PDFPlumberLoader as per your original code
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            if not docs:
                 logger.warning(f"PDFPlumberLoader returned no documents for: {file_path}")

            # Add source metadata (using basename is fine for local files)
            file_name = os.path.basename(file_path)
            for doc in docs:
                # Ensure metadata dictionary exists before access
                if not hasattr(doc, 'metadata'):
                     doc.metadata = {}
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_name
                # Optional: Add full path?
                # doc.metadata['full_path'] = file_path

            logger.info(f"Loaded {len(docs)} document(s) from PDF: {file_name} using PDFPlumber")
            return docs
        except Exception as e:
            # Log the error with traceback for debugging
            logger.error(f"Error loading PDF {file_path} with PDFPlumber: {e}", exc_info=True)
            return [] # Return empty list to allow script to continue

    @staticmethod
    # In data_processing.py

    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """Loads content from a text file, trying UTF-8 then UTF-16."""
        logger.info(f"Attempting to load TXT: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"Text file not found: {file_path}")
            return []

        docs = []
        try:
            # Try UTF-8 first
            logger.debug(f"Trying UTF-8 encoding for {file_path}")
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            logger.debug(f"Successfully loaded {file_path} with UTF-8")

        except UnicodeDecodeError:
            try:
                # If UTF-8 fails, try UTF-16
                logger.warning(f"UTF-8 decoding failed for {file_path}. Trying UTF-16 encoding.")
                loader = TextLoader(file_path, encoding='utf-16')
                docs = loader.load()
                logger.debug(f"Successfully loaded {file_path} with UTF-16")
            except Exception as e_utf16:
                # Catch errors during UTF-16 loading
                logger.error(f"Error loading text file {file_path} with UTF-16 fallback: {e_utf16}", exc_info=True)
                return [] # Failed to load with both encodings

        except Exception as e_utf8:
            # Catch other potential errors during UTF-8 loading (e.g., file permissions)
            logger.error(f"Error loading text file {file_path} with UTF-8: {e_utf8}", exc_info=True)
            return [] # Failed to load with UTF-8 for other reasons

        # --- Common processing after successful load ---
        if not docs:
            logger.warning(f"TextLoader returned no documents (file might be empty): {file_path}")
            # Return empty list if file loaded but was empty, allows script to continue normally
            return []

        file_name = os.path.basename(file_path)
        for doc in docs:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            if 'source' not in doc.metadata:
                doc.metadata['source'] = file_name

        logger.info(f"Loaded {len(docs)} document(s) from text file: {file_name}")
        return docs
    
    @staticmethod
    def load_docx(file_path: str) -> List[Document]:
        """Loads content from a DOCX file using Docx2txtLoader."""
        logger.info(f"Attempting to load DOCX: {file_path} using Docx2txtLoader")
        if not os.path.exists(file_path):
            logger.error(f"DOCX file not found: {file_path}")
            return []
        try:
            # Using Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            if not docs:
                 logger.warning(f"Docx2txtLoader returned no documents for: {file_path}")

            # Add source metadata
            file_name = os.path.basename(file_path)
            for doc in docs:
                if not hasattr(doc, 'metadata'):
                     doc.metadata = {}
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_name
                # doc.metadata['full_path'] = file_path # Optional

            logger.info(f"Loaded {len(docs)} document(s) from DOCX: {file_name} using Docx2txtLoader")
            return docs
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path} with Docx2txtLoader: {e}", exc_info=True)
            return []
    
        # Add methods for other data sources (CSV, JSON, Databases etc.)

# --- TextProcessor class remains the same ---
class TextProcessor:
    """Splits documents into chunks."""

    def __init__(self, chunk_size: int = settings.chunk_size, chunk_overlap: int = settings.chunk_overlap):
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info(f"Text splitter initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Splits a list of documents."""
        if not docs:
            logger.warning("No documents provided to split.")
            return []
        splits = self.text_splitter.split_documents(docs)
        logger.info(f"Split {len(docs)} documents into {len(splits)} chunks.")
        # Add chunk ID for better traceability
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = i
        return splits