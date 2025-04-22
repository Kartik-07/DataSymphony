# RAG_Project/Backend/rag_pipeline.py

import logging
import re
import json
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from retrieval import EnhancedRetriever
from generation import AnswerGenerator
from indexing import Indexer
from sql_processing import (
    generate_sql_query,
    clean_sql_string,
    get_rag_db_utility,
    get_uploads_db_utility,
    SQLDatabase
)
from config import settings

logger = logging.getLogger(__name__)

# --- RAG Response Model ---
class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: Optional[List[str]] = None

    @classmethod
    def from_docs(cls, answer: str, docs: List[Document]):
        sources_data = []
        if docs:
            for doc in docs:
                source_label = doc.metadata.get('identifier', doc.metadata.get('source', 'Unknown'))
                source_info = {
                    "source": source_label, "content_snippet": doc.page_content[:200] + "...",
                    "data_type": doc.metadata.get('data_type'), "source_type": doc.metadata.get('source_type'),
                    "target_database": doc.metadata.get('target_database'), "target_table_name": doc.metadata.get('target_table_name'),
                    "full_metadata": doc.metadata
                }
                source_info = {k: v for k, v in source_info.items() if v is not None}
                sources_data.append(source_info)
        final_answer = answer if isinstance(answer, str) else "Error processing generation result."
        return cls(answer=final_answer, sources=sources_data, suggestions=None)


# --- RAG Pipeline Class ---
class RAGPipeline:
    """
    Orchestrates the RAG query process with LLM-driven routing
    (SQL vs. Vector Store) and suggestion generation. Uses SYNCHRONOUS operations.
    """
    def __init__(self, indexer: Indexer):
        self.retriever_wrapper = EnhancedRetriever(indexer)
        self.generator = AnswerGenerator()
        self.rag_db_utility: Optional[SQLDatabase] = get_rag_db_utility()
        self.uploads_db_utility: Optional[SQLDatabase] = get_uploads_db_utility()
        try:
            self.light_llm = ChatGoogleGenerativeAI(
                model=settings.light_llm_model_name,
                temperature=0.1,
            )
            logger.info(f"Pipeline Light LLM initialized: {settings.light_llm_model_name}")
        except Exception as e:
             logger.error(f"Failed to init Light LLM ({settings.light_llm_model_name}): {e}", exc_info=True)
             self.light_llm = None
        logger.info("RAG Pipeline initialized (sync operations).")
        if not self.rag_db_utility: logger.warning("RAG_DB utility unavailable.")
        if not self.uploads_db_utility: logger.warning("RAG_DB_UPLOADS utility unavailable.")
        if not self.light_llm: logger.warning("Light LLM unavailable.")

    def _format_metadata_for_router(self, docs: List[Document]) -> str:
        # (Keep this helper function as defined previously)
        if not docs: return "No potentially relevant data sources found."
        formatted_sources = []
        for i, doc in enumerate(docs):
            meta = doc.metadata; source_id = meta.get('identifier', f'unknown_{i}')
            data_type = meta.get('data_type', 'unknown'); source_type = meta.get('source_type', 'unknown')
            doc_summary = doc.page_content
            source_info = f"Source {i+1}:\n  ID: {source_id}\n  Type: {data_type} ({source_type})\n  Summary: {doc_summary}\n"
            if data_type == 'structured' and 'columns' in meta and isinstance(meta['columns'], list):
                 col_names = [str(col.get('column', '?')) for col in meta['columns'][:15]]
                 source_info += f"  Columns: {', '.join(col_names)}{'...' if len(meta['columns']) > 15 else ''}\n"
            elif data_type == 'unstructured' and 'keywords' in meta and isinstance(meta['keywords'], list):
                 source_info += f"  Keywords: {', '.join(meta['keywords'][:10])}\n"
            formatted_sources.append(source_info.strip())
        return "\n---\n".join(formatted_sources)

    # --- SWITCHED TO SYNC ---
    def _route_query_llm(self, query_text: str) -> tuple[str, Optional[Dict[str, Any]]]:
        """Uses the Light LLM (synchronously) to decide the best data source."""
        logger.info(f"Routing query (sync): '{query_text[:50]}...'")
        decision_type = 'NONE'; target_metadata = None
        if not self.light_llm: logger.error("Routing LLM unavailable."); return 'VECTOR_STORE', None # Fallback

        try:
            # 1. Retrieve metadata candidates (synchronously)
            base_retriever: BaseRetriever = self.retriever_wrapper.base_retriever
            try:
                metadata_retriever = base_retriever.vectorstore.as_retriever(search_kwargs={'k': 7})
                # Use synchronous invoke
                retrieved_metadata_docs = metadata_retriever.invoke(query_text)
                logger.info(f"Retrieved {len(retrieved_metadata_docs)} metadata candidates (sync).")
            except Exception as retr_err: logger.error(f"Metadata retrieval failed (sync): {retr_err}"); retrieved_metadata_docs = []
            if not retrieved_metadata_docs: logger.warning("No metadata docs found."); return 'NONE', None

            # 2. Format metadata
            formatted_sources_str = self._format_metadata_for_router(retrieved_metadata_docs)

            # 3. Prepare Router Prompt (keep the detailed prompt)
            router_system_prompt = """
You are an expert data routing assistant. Your task is to analyze the user's query and determine the single best data source to answer it from the list provided below.

Available Data Sources:
---
{available_sources}
---

Instructions:
1. Carefully examine the User Query.
2. Review each Available Data Source, paying attention to: ID, Type, Summary, Columns (structured), Keywords (unstructured).
3. Make a Decision:
    * Choose SQL: If the query asks for specific data points, calculations, filtering, or aggregations matching a structured source's Columns and Summary. Respond with `SQL:{{Source ID}}` (e.g., `SQL:Course_Completion.xlsx`).
    * Choose VECTOR_STORE: If the query is general, asks for explanations, concepts, comparisons, or matches unstructured sources, or if no single structured source is perfect. Respond with `VECTOR_STORE`.
    * Choose NONE: If no source seems relevant. Respond with `NONE`.
4. Output Format: Respond ONLY with the chosen decision (`SQL:{{Source ID}}`, `VECTOR_STORE`, or `NONE`).
            """
            router_prompt_messages = [ SystemMessage(content=router_system_prompt.format(available_sources=formatted_sources_str)),
                                      HumanMessage(content=f"User Query:\n\"{query_text}\"\n\nDecision:") ]

            # 4. Invoke Router LLM (synchronously)
            router_chain = self.light_llm | StrOutputParser()
            raw_decision = router_chain.invoke(router_prompt_messages) # Use sync invoke
            decision = raw_decision.strip()
            logger.info(f"LLM Router Decision (sync): {decision}")

            # 5. Parse Decision (logic remains the same)
            if decision == 'VECTOR_STORE': decision_type = 'VECTOR_STORE'
            elif decision == 'NONE': decision_type = 'NONE'
            elif decision.startswith('SQL:'):
                target_identifier = decision.split(':', 1)[1].strip()
                chosen_doc = next((doc for doc in retrieved_metadata_docs if doc.metadata.get('identifier') == target_identifier), None)
                if chosen_doc:
                    target_metadata = chosen_doc.metadata; db_key = 'NONE'
                    target_db_name = target_metadata.get('target_database')
                    if target_db_name == 'RAG_DB_UPLOADS' and self.uploads_db_utility: db_key = 'UPLOADS'
                    elif target_db_name == 'RAG_DB' and self.rag_db_utility: db_key = 'MAIN'
                    if db_key != 'NONE': decision_type = f"SQL:{db_key}:{target_identifier}"; logger.info(f"Routing confirmed (sync) to {decision_type}.")
                    else: logger.warning(f"SQL source '{target_identifier}' chosen but DB utility '{target_db_name}' unavailable. Fallback NONE."); decision_type = 'NONE'; target_metadata = None
                else: logger.warning(f"SQL source '{target_identifier}' chosen but metadata not found. Fallback VECTOR_STORE."); decision_type = 'VECTOR_STORE'
            else: logger.warning(f"Unexpected router decision: '{decision}'. Fallback VECTOR_STORE."); decision_type = 'VECTOR_STORE'
        except Exception as e: logger.error(f"LLM routing error (sync): {e}"); decision_type = 'VECTOR_STORE' # Fallback
        return decision_type, target_metadata

    # --- SWITCHED TO SYNC ---
    def _generate_suggestions(self, user_query: str, ai_answer: str) -> Optional[List[str]]:
        """Generates follow-up suggestions using the light LLM (synchronously)."""
        if not user_query or not ai_answer or not self.light_llm: return None
        logger.info("Generating suggestions (sync)...")
        try:
            suggestion_prompt = f"""
            Given the user's query and the AI's answer, suggest 3 concise and relevant follow-up questions or actions.
            Phrase them as if the user is asking.
            Respond ONLY with a valid JSON list of strings, like ["Suggestion 1", "Suggestion 2", "Suggestion 3"].

            User Query: "{user_query}"
            AI Answer: "{ai_answer}"
            Suggested follow-up questions/actions (JSON list):
            """
            messages = [
                 SystemMessage(content="You suggest relevant follow-up questions/actions."),
                 HumanMessage(content=suggestion_prompt)
            ]
            suggestion_chain = self.light_llm | StrOutputParser()
            # Use synchronous invoke with the list of messages
            raw_suggestions_output = suggestion_chain.invoke(messages)
            logger.debug(f"Raw suggestions output (sync): {raw_suggestions_output}")
            try: # Parse JSON
                cleaned_output = re.sub(r"```(?:json)?\s*(.*)\s*```", r"\1", raw_suggestions_output, flags=re.DOTALL).strip()
                suggestions = json.loads(cleaned_output)
                if isinstance(suggestions, list) and all(isinstance(s, str) for s in suggestions):
                    logger.info(f"Generated suggestions (sync): {suggestions[:3]}"); return suggestions[:3]
                else: logger.warning(f"Suggestion output not list of strings: {cleaned_output}"); return None
            except json.JSONDecodeError: # Fallback parsing
                logger.warning(f"Failed to parse suggestions JSON: {raw_suggestions_output}")
                extracted = re.findall(r'"([^"]+)"', raw_suggestions_output)
                if extracted: logger.info(f"Extracted suggestions via regex: {extracted[:3]}"); return extracted[:3]
                return None
        except Exception as e: logger.error(f"Suggestion generation error (sync): {e}"); return None

    # --- SWITCHED TO SYNC ---
    def query(self, query_text: str) -> RAGResponse:
        """Executes the RAG pipeline synchronously with LLM routing."""
        logger.info(f"Processing query (sync): '{query_text[:50]}...'")
        try:
            # Call synchronous router
            decision, target_metadata = self._route_query_llm(query_text)
        except Exception as route_err:
            logger.error(f"Fatal routing error (sync): {route_err}", exc_info=True)
            return RAGResponse(answer="Internal error during query routing.", sources=[], suggestions=None)

        suggestions = None; final_answer = ""; sources_for_response = []

        # --- Execute Based on Route (Sync) ---
        if decision.startswith('SQL:'):
            parts = decision.split(':', 2); db_key = parts[1]; target_identifier = parts[2]
            target_db_utility = self.uploads_db_utility if db_key == 'UPLOADS' else self.rag_db_utility
            if target_db_utility and target_metadata:
                db_name = getattr(target_db_utility._engine.url, 'database', 'Unknown DB')
                table_name = target_metadata.get("target_table_name", target_identifier)
                logger.info(f"LLM routed to SQL (sync). DB: {db_name}, Table: {table_name}")
                try: # SQL Generation and Execution (Sync)
                    raw_sql = generate_sql_query(query_text, target_db_utility, target_metadata) # Assumed sync
                    if not raw_sql: raise ValueError("SQL generation failed.")
                    cleaned_sql = clean_sql_string(raw_sql)
                    if not cleaned_sql: raise ValueError("Generated SQL invalid.")
                    logger.info(f"Executing SQL (sync): {cleaned_sql}")
                    db_execution_result = target_db_utility.run(cleaned_sql) # Sync execution
                    logger.info(f"SQL Result (sync): {str(db_execution_result)[:500]}...")
                    final_answer = f"Result from '{table_name}':\n{db_execution_result}"
                    sources_for_response = [{"source": f"DB: {db_name}/{table_name}", "query": cleaned_sql, "content_snippet": str(db_execution_result)[:200]+"...", "full_metadata": target_metadata}]
                except Exception as e: logger.error(f"SQL processing failed (sync): {e}"); final_answer = f"Failed structured query: {e}"
            else: logger.error(f"SQL route missing DB utility/metadata. Fallback NONE."); decision = 'NONE'

        elif decision == 'VECTOR_STORE':
            logger.info("LLM routed to Vector Store (sync).")
            try: # Vector Retrieval and Generation (Sync)
                # Use synchronous invoke
                retrieved_docs = self.retriever_wrapper.final_retriever.invoke(query_text)
                if not retrieved_docs: logger.warning("No relevant docs found (sync)."); final_answer = "Could not find relevant information."
                else:
                    final_answer = self.generator.generate_answer(query_text, retrieved_docs) # Sync call
                    response_obj = RAGResponse.from_docs(answer=final_answer, docs=retrieved_docs)
                    sources_for_response = response_obj.sources
            except Exception as e: logger.error(f"RAG processing failed (sync): {e}"); final_answer = f"Error during retrieval/generation: {e}"

        # Handle NONE case
        if decision == 'NONE':
             logger.warning(f"LLM router found no relevant source (sync): '{query_text[:50]}...'")
             final_answer = "No relevant data source found to answer the query."
             sources_for_response = []

        # --- Generate Suggestions (Sync) ---
        if decision != 'NONE' and final_answer:
             try: suggestions = self._generate_suggestions(query_text, final_answer) # Sync call
             except Exception as sugg_err: logger.error(f"Suggestion generation failed (sync): {sugg_err}"); suggestions = None

        # --- Construct Final Response ---
        if not isinstance(final_answer, str): final_answer = "Internal error formulating answer."
        logger.info(f"Query processing complete (sync): '{query_text[:50]}...'")
        return RAGResponse(answer=final_answer, sources=sources_for_response, suggestions=suggestions)