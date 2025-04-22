import logging
import sys
import psycopg2
from psycopg2 import pool

# --- Existing setup_logging function ---
def setup_logging(level=logging.INFO):
    """Sets up basic logging."""
    # Avoid adding duplicate handlers if called multiple times
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()

    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    logger.setLevel(level)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    # Optional: File handler
    # file_handler = logging.FileHandler("rag_system.log")
    # file_handler.setFormatter(log_formatter)
    # logger.addHandler(file_handler)

    logging.info("Logging configured.")

# Call setup_logging when the module is imported or explicitly in main.py
# setup_logging() # Or call from main

# --- Optional: Postgres Pool (only needed for DIRECT SQL outside PGVector/SQLAlchemy) ---
class PostgresConnectionPool:
    """Minimal wrapper around psycopg2 pool for direct connections if needed."""
    _pool = None

    @classmethod
    def initialize(cls, dsn: str, minconn: int = 1, maxconn: int = 5):
        if cls._pool is None:
            try:
                cls._pool = pool.SimpleConnectionPool(minconn, maxconn, dsn=dsn)
                logging.info(f"Initialized PostgreSQL connection pool (min={minconn}, max={maxconn}).")
            except Exception as e:
                logging.error(f"Failed to create PostgreSQL connection pool: {e}", exc_info=True)
                cls._pool = None # Ensure it's None on failure
        return cls._pool is not None

    @classmethod
    def getconn(cls):
        if cls._pool:
            return cls._pool.getconn()
        raise ConnectionError("Postgres connection pool is not initialized.")

    @classmethod
    def putconn(cls, conn):
         if cls._pool:
             cls._pool.putconn(conn)

    @classmethod
    def closeall(cls):
         if cls._pool:
             cls._pool.closeall()
             logging.info("Closed PostgreSQL connection pool.")
             cls._pool = None

# --- Helper to get DSN for psycopg2 from SQLAlchemy URL ---
def get_psycopg2_dsn(sqlalchemy_url: str) -> str:
    """Converts a SQLAlchemy URL to a psycopg2 DSN string."""
    from urllib.parse import urlparse
    parsed = urlparse(sqlalchemy_url)
    dsn = f"dbname={parsed.path[1:]} user={parsed.username} password={parsed.password} host={parsed.hostname} port={parsed.port}"
    return dsn