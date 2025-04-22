import logging
import sys
from typing import List, Optional
import psycopg2 # Needed for direct check/setup

# Use specific PGVector import if available and preferred
# from langchain_community.vectorstores import PGVector
from langchain_postgres.vectorstores import PGVector

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import settings
from utils import PostgresConnectionPool, get_psycopg2_dsn # Optional Pool import

logger = logging.getLogger(__name__)

def ensure_pgvector_setup():
    """
    Ensures the vector extension exists in the DB using a direct psycopg2 connection.
    PGVector Langchain integration handles table creation.
    """
    logger.info("Checking for PGVector extension...")
    try:
        # Get a direct connection using psycopg2 DSN format
        dsn = get_psycopg2_dsn(settings.postgres_url)
        # Initialize pool if not already done (e.g., in API startup)
        # if PostgresConnectionPool._pool is None:
        #     PostgresConnectionPool.initialize(dsn)

        # conn = PostgresConnectionPool.getconn() # Use pool if initialized
        conn = psycopg2.connect(dsn) # Direct connection for setup
        conn.autocommit = True # Use autocommit for DDL commands

        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("PGVector extension check complete (created if not exists).")

    except psycopg2.Error as e:
        logger.error(f"Database error during PGVector setup check: {e}", exc_info=True)
        # Decide if this is fatal - maybe the DB isn't ready yet?
        # sys.exit("Exiting: Failed to ensure PGVector extension.") # Uncomment to make it fatal
        raise ConnectionError(f"Failed to connect/setup PGVector: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during PGVector setup check: {e}", exc_info=True)
        raise
    finally:
        if 'conn' in locals() and conn is not None:
             conn.close() # Close direct connection
             # PostgresConnectionPool.putconn(conn) # Use pool if initialized


class Indexer:
    """Handles embedding generation and PGVector interactions."""

    def __init__(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={'device': 'cpu'}, # Or 'cuda' if GPU available
                encode_kwargs={'normalize_embeddings': True} # Normalize for cosine similarity
            )
            logger.info(f"Initialized HuggingFaceEmbeddings with model: {settings.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{settings.embedding_model_name}': {e}", exc_info=True)
            raise

        # Initialize PGVector client
        try:
             # Ensure the vector dimension is passed if needed by the constructor
             # or relies on table schema which ensure_pgvector_setup doesn't create
             self.vector_store = PGVector(
                 connection=settings.postgres_url,
                 embeddings=self.embeddings,
                 collection_name=settings.collection_name,
                 # driver=settings.vector_store_driver, # May not be needed if using SQLAlchemy URL
                 # pre_delete_collection=False, # Default
                 use_jsonb=True # Store metadata in JSONB
             )
             logger.info(f"PGVector client initialized. Collection: {settings.collection_name}")
             # Note: PGVector Langchain integration often creates the table on first use if it doesn't exist.
             # The `ensure_pgvector_setup` only guarantees the EXTENSION.
             # You might need to explicitly create the table with the correct embedding dimension
             # if the Langchain integration doesn't handle it automatically based on embedding_dimension.

        except ImportError as e:
             logger.error(f"Failed to import PGVector dependencies. Install 'langchain-postgres'. Error: {e}")
             raise
        except Exception as e:
             logger.error(f"Failed to initialize PGVector: {e}", exc_info=True)
             # Log the connection string without credentials for debugging
             safe_conn_string = settings.postgres_url.split('@')[-1] # Basic obfuscation
             logger.error(f"Check your POSTGRES_URL connection string format and DB accessibility (trying to connect to ...@{safe_conn_string}).")
             raise

    def index_documents(self, docs: List[Document], ids: Optional[List[str]] = None):
        """Embeds and stores documents in PGVector."""
        if not docs:
            logger.warning("No documents provided for indexing.")
            return

        try:
            # PGVector's add_documents handles embedding and insertion
            added_ids = self.vector_store.add_documents(docs, ids=ids)
            logger.info(f"Successfully added/updated {len(docs)} documents to PGVector collection '{settings.collection_name}'. IDs: {added_ids[:5]}...")
        except Exception as e:
            logger.error(f"Failed to index documents into PGVector: {e}", exc_info=True)
            # Consider more specific error handling based on DB exceptions
            raise

    def delete_documents(self, ids: List[str]) -> bool:
        """Deletes documents from the vector store by their IDs."""
        if not ids:
            logger.warning("No IDs provided for deletion.")
            return False
        try:
            self.vector_store.delete(ids=ids)
            logger.info(f"Attempted deletion of {len(ids)} documents from collection '{settings.collection_name}'.")
            return True
        except Exception as e:
             logger.error(f"Failed to delete documents with IDs {ids[:5]}: {e}", exc_info=True)
             return False


    def get_retriever(self, search_type: str = "similarity", k: int = settings.retriever_k):
        """Gets a retriever instance from the PGVector store."""
        try:
            retriever = self.vector_store.as_retriever(
                search_type=search_type, # similarity, similarity_score_threshold, mmr
                search_kwargs={'k': k}
            )
            logger.info(f"PGVector retriever created with search_type={search_type}, k={k}")
            return retriever
        except Exception as e:
            logger.error(f"Failed to create PGVector retriever: {e}", exc_info=True)
            raise