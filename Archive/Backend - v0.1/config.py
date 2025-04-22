import os
import logging
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file before defining settings
load_dotenv()
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # LangSmith
    langchain_tracing_v2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    langchain_endpoint: str | None = os.getenv("LANGCHAIN_ENDPOINT")
    langchain_api_key: str | None = os.getenv("LANGCHAIN_API_KEY")

    # LLM/Embedding Providers
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "default_google_key")
    cohere_api_key: str | None = os.getenv("COHERE_API_KEY")

    # Database
    postgres_url: str = os.getenv("POSTGRES_URL", "postgresql+psycopg2://dummy:dummy@localhost:5432/dummy")
    vector_store_driver: str = os.getenv("VECTOR_STORE_DRIVER", "psycopg2")

    # Models & Behavior
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", 768)) # Defaulting to mpnet dimension
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "gemini-pro")
    collection_name: str = os.getenv("COLLECTION_NAME", "rag_collection")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))
    retriever_k: int = int(os.getenv("RETRIEVER_K", 10))
    reranker_top_n: int = int(os.getenv("RERANKER_TOP_N", 3))
    use_cohere_rerank: bool = os.getenv("USE_COHERE_RERANK", "True").lower() == "true"

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'

# Instantiate settings
settings = Settings()

# --- Environment Variable Setup & Validation ---
if not settings.google_api_key or settings.google_api_key == "default_google_key":
    raise ValueError("GOOGLE_API_KEY must be set in the .env file.")
if "dummy" in settings.postgres_url:
     logger.warning("POSTGRES_URL is set to a dummy value. Please configure it in the .env file.")
     # raise ValueError("POSTGRES_URL must be configured in the .env file.") # Uncomment to make it mandatory

if settings.use_cohere_rerank and not settings.cohere_api_key:
    logger.warning("USE_COHERE_RERANK is True but COHERE_API_KEY is not set. Reranking will be disabled.")
    settings.use_cohere_rerank = False # Disable if key is missing

# Set environment variables for LangChain/SDKs that might read them directly
os.environ['GOOGLE_API_KEY'] = settings.google_api_key # For Google SDK
if settings.langchain_api_key:
    os.environ['LANGCHAIN_TRACING_V2'] = settings.langchain_tracing_v2
    os.environ['LANGCHAIN_ENDPOINT'] = str(settings.langchain_endpoint)
    os.environ['LANGCHAIN_API_KEY'] = settings.langchain_api_key
if settings.cohere_api_key and settings.use_cohere_rerank:
     os.environ['COHERE_API_KEY'] = settings.cohere_api_key

logger.info(f"Loaded settings: LLM={settings.llm_model_name}, Embedding={settings.embedding_model_name} ({settings.embedding_dimension}d)")
logger.info(f"Vector Store: PGVector (Collection: {settings.collection_name}), Reranker Enabled: {settings.use_cohere_rerank}")