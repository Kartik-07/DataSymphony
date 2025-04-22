import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from retrieval import EnhancedRetriever
from generation import AnswerGenerator
from indexing import Indexer
# Import the specific functions/objects needed from sql_processing
from sql_processing import is_structured_query, get_sql_database_utility, clean_sql_string, generate_sql_query # Updated imports

logger = logging.getLogger(__name__)

# RAGResponse class remains the same
class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    # from_docs classmethod remains the same

    @classmethod
    def from_docs(cls, answer: str, docs: List[Document]):
        sources_data = []
        if docs:
            for doc in docs:
                 source_info = {
                     "metadata": doc.metadata,
                     "content_snippet": doc.page_content[:200] + "..."
                 }
                 sources_data.append(source_info)
        final_answer = answer if isinstance(answer, str) else "Error processing generation result."
        return cls(answer=final_answer, sources=sources_data)


class RAGPipeline:
    """Orchestrates the RAG query process with manual SQL generation and execution."""

    def __init__(self, indexer: Indexer):
        self.retriever = EnhancedRetriever(indexer)
        self.generator = AnswerGenerator()
        self.db_utility = get_sql_database_utility() # Get the DB utility
        logger.info("RAG Pipeline initialized.")

    def query(self, query_text: str) -> RAGResponse:
        """Executes the RAG pipeline or manual SQL flow based on query structure."""
        logger.info(f"Processing query: '{query_text[:50]}...'")

        # --- Routing Logic ---
        if is_structured_query(query_text):
            logger.info("Routing query to manual SQL flow.")
            raw_sql = None
            cleaned_sql = None
            db_execution_result = None

            if not self.db_utility:
                logger.error("SQL database utility is not available.")
                return RAGResponse(answer="Unable to process structured query due to configuration error.", sources=[])

            try:
                # 1. Generate SQL query string
                raw_sql = generate_sql_query(query_text)
                if not raw_sql:
                    raise ValueError("Failed to generate SQL query from the language model.")

                # 2. Clean the generated SQL query
                cleaned_sql = clean_sql_string(raw_sql)
                if not cleaned_sql: # Check if cleaning resulted in empty string
                     logger.error(f"SQL query became empty after cleaning raw query: '{raw_sql}'")
                     raise ValueError("Generated SQL query is invalid after cleaning.")

                # 3. Execute the Cleaned SQL Query
                logger.info(f"Executing manually cleaned SQL: {cleaned_sql}")
                db_execution_result = self.db_utility.run(cleaned_sql)
                logger.info(f"Manual SQL Execution Result: {db_execution_result}")

                # 4. Format the final answer (Using the direct result for now)
                # You could potentially pass the result back to the LLM for summarization if needed
                final_answer = str(db_execution_result)

                # Prepare response
                sources = [{"metadata": {"source": "Structured Database"}, "content_snippet": f"Executed SQL: {cleaned_sql}"}]
                return RAGResponse(answer=final_answer, sources=sources)

            except Exception as e:
                logger.error(f"Manual SQL processing failed: {e}", exc_info=True)
                error_message = f"Failed to execute structured query. Error: {str(e)}"
                # Include SQL if available, helps debugging
                if cleaned_sql: error_message += f" [Cleaned SQL: {cleaned_sql}]"
                elif raw_sql: error_message += f" [Raw SQL attempt: {raw_sql[:100]}...]"
                return RAGResponse(answer=error_message, sources=[])

        # --- Vector Store RAG Path (if not structured) ---
        else:
            logger.info("Routing query to Vector Store (RAG).")
            # ... (RAG logic remains the same) ...
            retrieved_docs = self.retriever.retrieve_documents(query_text)
            if not retrieved_docs:
                logger.warning("No relevant documents found via RAG.")
                return RAGResponse(answer="Could not find relevant information in documents to answer the query.", sources=[])
            answer = self.generator.generate_answer(query_text, retrieved_docs)
            response = RAGResponse.from_docs(answer=answer, docs=retrieved_docs)
            logger.info(f"RAG query processed successfully.")
            return response