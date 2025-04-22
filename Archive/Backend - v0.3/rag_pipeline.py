# RAG_Project/MY_RAG/Backend/rag_pipeline.py

import logging
import re
import json
from typing import List, Dict, Any, Optional

# --- Pydantic and Langchain Core Imports ---
from pydantic import BaseModel, Field # Keep local RAGResponse definition for now based on original
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate # Added

# --- Local Module Imports ---
from config import settings
from retrieval import EnhancedRetriever # From original code
from generation import AnswerGenerator # From original code - Needs modification
from indexing import Indexer # From original code
from sql_processing import ( # From original code
    generate_sql_query,
    clean_sql_string,
    get_rag_db_utility,
    get_uploads_db_utility,
    SQLDatabase
)
# --- Import the prompt *creation function* and history formatter ---
from prompts import create_rag_prompt_template, format_chat_history, FALLBACK_PROMPT, FALLBACK_PROMPT_WITH_HISTORY
# Note: The generator will use create_rag_prompt_template internally now

# --- Import ChatMessage for type hinting ---
# (Keep the robust import attempt)
try:
    from models import ChatMessage
except ImportError:
    logging.warning("Could not import ChatMessage from models.py in rag_pipeline.py. Using basic definition.")
    from pydantic import BaseModel as PydanticBaseModel # Alias to avoid conflict
    from datetime import datetime
    class ChatMessage(PydanticBaseModel): # Basic definition if import fails
        sender: str
        text: str
        timestamp: Optional[datetime] = None
        sources: Optional[List[Dict[str, Any]]] = None
        error: Optional[bool] = None

logger = logging.getLogger(__name__)

# --- RAG Response Model (Defined locally as in original) ---
# Consider moving to a shared models.py file eventually
class RAGResponse(BaseModel):
    """Pydantic model for the final response structure."""
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: Optional[List[str]] = None

    @staticmethod
    def format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
        """Helper to format sources from documents, ensuring uniqueness."""
        # This static method remains the same as in your original code
        sources_data = []
        if not docs: return sources_data
        seen_identifiers = set()
        for doc in docs:
            # Prioritize 'identifier', fallback to 'source', then generate unique unknown
            identifier = doc.metadata.get('identifier', doc.metadata.get('source'))
            if not identifier:
                # Use hash of content snippet for more deterministic unknown ID
                content_hash = hash(doc.page_content[:50])
                identifier = f"unknown_{content_hash}_{len(seen_identifiers)}"

            if identifier in seen_identifiers: continue
            seen_identifiers.add(identifier)

            source_label = doc.metadata.get('file_name', identifier) # Prefer file_name if available
            source_type_flag = doc.metadata.get('type', 'vector') # Default type

            source_info = {
                "source": source_label,
                "type": source_type_flag,
                "content_snippet": doc.page_content[:200] + "...",
                "relevance_score": doc.metadata.get('relevance_score'),
                "identifier": identifier, # Pass identifier for potential frontend use
                # Pass specific metadata if useful
                "page": doc.metadata.get('page'),
                "summary_id": doc.metadata.get('summary_id'),
                "uploaded_by": doc.metadata.get('uploaded_by'),
            }
            # Only include keys with non-None values
            sources_data.append({k: v for k, v in source_info.items() if v is not None})
        return sources_data


# --- RAG Pipeline Class ---
class RAGPipeline:
    """
    Orchestrates RAG: Routing -> Score Check -> Generation + Sufficiency Check -> Fallback -> Suggestions.
    Uses generator's flag for supplemental fallback decision and incorporates chat history.
    """
    def __init__(self, indexer: Indexer):
        # Initialization largely remains the same as your original code
        self.retriever_wrapper = EnhancedRetriever(indexer)
        # Ensure this generator instance matches the one modified to handle history
        self.generator = AnswerGenerator() # This generator NEEDS modification (see notes below)
        self.rag_db_utility: Optional[SQLDatabase] = get_rag_db_utility()
        self.uploads_db_utility: Optional[SQLDatabase] = get_uploads_db_utility()
        try:
            self.light_llm = ChatGoogleGenerativeAI(model=settings.light_llm_model_name, temperature=0.1)
            logger.info(f"Pipeline Light LLM (Routing/Suggestions) initialized: {settings.light_llm_model_name}")
        except Exception as e:
            logger.error(f"Failed to init Pipeline Light LLM: {e}", exc_info=True)
            self.light_llm = None
        logger.info("RAG Pipeline initialized (sync operations).")
        # Availability logging remains the same
        if not self.rag_db_utility: logger.warning("RAG_DB utility unavailable.")
        if not self.uploads_db_utility: logger.warning("RAG_DB_UPLOADS utility unavailable.")
        if not self.light_llm: logger.warning("Pipeline Light LLM (Routing/Suggestions) unavailable.")
        if not self.generator.llm: logger.warning("Main Generator LLM unavailable.")
        # Assuming light_llm in generator is for sufficiency check
        if not self.generator.light_llm: logger.warning("Generator's Sufficiency Check LLM unavailable.")

    # --- Helper methods (_format_metadata_for_router, _route_query_llm, _generate_suggestions) ---
    # Keep these methods exactly as in your original code
    def _format_metadata_for_router(self, docs: List[Document]) -> str:
        # ... (Same as original code) ...
        if not docs: return "No potentially relevant data sources found."
        formatted_sources = []
        for i, doc in enumerate(docs):
            meta = doc.metadata; source_id = meta.get('identifier', f'unknown_{i}')
            data_type = meta.get('data_type', 'unknown'); source_type = meta.get('source_type', 'unknown')
            # Use page_content as summary if 'summary' field doesn't exist
            doc_summary = meta.get('summary', doc.page_content[:300]) # Limit summary length
            source_info = f"Source {i+1}:\n  ID: {source_id}\n  Type: {data_type} ({source_type})\n  Summary: {doc_summary}\n"
            if data_type == 'structured' and 'columns' in meta and isinstance(meta['columns'], list):
                 col_names = [str(col.get('column', '?')) for col in meta['columns'][:15]]; source_info += f"  Columns: {', '.join(col_names)}{'...' if len(meta['columns']) > 15 else ''}\n"
            elif data_type == 'unstructured' and 'keywords' in meta and isinstance(meta['keywords'], list):
                 source_info += f"  Keywords: {', '.join(meta['keywords'][:10])}\n"
            formatted_sources.append(source_info.strip())
        return "\n---\n".join(formatted_sources)

    def _route_query_llm(self, query_text: str) -> tuple[str, Optional[Dict[str, Any]]]:
        # ... (Same routing logic as original code) ...
        logger.info(f"Routing query (sync): '{query_text[:50]}...'")
        decision_type = 'NONE'; target_metadata = None
        if not self.light_llm: logger.error("Routing LLM unavailable. Falling back to VECTOR_STORE."); return 'VECTOR_STORE', None
        try:
            base_retriever: BaseRetriever = self.retriever_wrapper.base_retriever
            try:
                # Retrieve metadata candidates - adjust k if needed
                metadata_retriever = base_retriever.vectorstore.as_retriever(search_kwargs={'k': 7})
                retrieved_metadata_docs = metadata_retriever.invoke(query_text)
                logger.info(f"Retrieved {len(retrieved_metadata_docs)} metadata candidates for routing (sync).")
            except Exception as retr_err:
                 logger.error(f"Metadata retrieval for routing failed: {retr_err}", exc_info=True)
                 retrieved_metadata_docs = [] # Continue without candidates

            # If no candidates, default to VECTOR_STORE (or NONE if preferred)
            if not retrieved_metadata_docs:
                logger.warning("No metadata docs found during routing. Decision: VECTOR_STORE.")
                return 'VECTOR_STORE', None

            formatted_sources_str = self._format_metadata_for_router(retrieved_metadata_docs)
            # Using the same router prompt template as original
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
            router_prompt_messages = [ SystemMessage(content=router_system_prompt.format(available_sources=formatted_sources_str)), HumanMessage(content=f"User Query:\n\"{query_text}\"\n\nDecision:") ]
            router_chain = self.light_llm | StrOutputParser()
            raw_decision = router_chain.invoke(router_prompt_messages); decision = raw_decision.strip()
            logger.info(f"LLM Router Decision (sync): {decision}")

            # Parse Decision (same logic as original)
            if decision == 'VECTOR_STORE': decision_type = 'VECTOR_STORE'
            elif decision == 'NONE': decision_type = 'NONE'
            elif decision.startswith('SQL:'):
                target_identifier = decision.split(':', 1)[1].strip()
                chosen_doc = next((doc for doc in retrieved_metadata_docs if doc.metadata.get('identifier') == target_identifier), None)
                if chosen_doc:
                    target_metadata = chosen_doc.metadata; db_key = 'NONE'
                    target_db_name = target_metadata.get('target_database') # Assumes this metadata exists
                    if target_db_name == 'RAG_DB_UPLOADS' and self.uploads_db_utility: db_key = 'UPLOADS'
                    elif target_db_name == 'RAG_DB' and self.rag_db_utility: db_key = 'MAIN'
                    # Check if DB utility is available
                    if db_key != 'NONE':
                        decision_type = f"SQL:{db_key}:{target_identifier}"
                        # Add table name to metadata if not present, needed for SQL generation
                        if 'target_table_name' not in target_metadata:
                            target_metadata['target_table_name'] = target_identifier # Assuming identifier is table name
                    else:
                        logger.warning(f"SQL source '{target_identifier}' chosen but required DB utility (for {target_db_name}) unavailable. Fallback NONE.")
                        decision_type = 'NONE'
                        target_metadata = None # Clear metadata if DB unavailable
                else:
                    logger.warning(f"SQL source '{target_identifier}' chosen by router but metadata not found in retrieved docs. Fallback VECTOR_STORE.")
                    decision_type = 'VECTOR_STORE'
            else:
                 logger.warning(f"Unexpected router decision format: '{decision}'. Fallback VECTOR_STORE.")
                 decision_type = 'VECTOR_STORE'
        except Exception as e:
             logger.error(f"LLM routing step failed: {e}", exc_info=True)
             decision_type = 'VECTOR_STORE' # Fallback on error

        logger.info(f"Final routing decision: {decision_type}")
        return decision_type, target_metadata

    def _generate_suggestions(self, user_query: str, ai_answer: str) -> Optional[List[str]]:
         # ... (Same suggestion logic as original code) ...
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
            messages = [ SystemMessage(content="You suggest relevant follow-up questions/actions based on the conversation."), HumanMessage(content=suggestion_prompt) ]
            suggestion_chain = self.light_llm | StrOutputParser()
            raw_suggestions_output = suggestion_chain.invoke(messages)
            logger.debug(f"Raw suggestions output: {raw_suggestions_output}")
            try: # Parse JSON robustly (same logic)
                # More robust extraction if LLM adds markdown/text
                match = re.search(r"\[.*\]", raw_suggestions_output, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    suggestions = json.loads(json_str)
                    if isinstance(suggestions, list) and all(isinstance(s, str) for s in suggestions):
                         logger.info(f"Generated suggestions: {suggestions[:3]}")
                         return suggestions[:3]
                    else:
                         logger.warning(f"Parsed JSON not list of strings: {json_str}")
                         return None
                else:
                     logger.warning(f"Could not find JSON list in suggestions output: {raw_suggestions_output}")
                     # Fallback regex findall if no proper list found
                     extracted = re.findall(r'"([^"]+)"', raw_suggestions_output);
                     if extracted:
                         logger.info(f"Extracted suggestions via regex: {extracted[:3]}");
                         return extracted[:3]
                     return None

            except json.JSONDecodeError as json_err:
                 logger.warning(f"Failed to parse suggestions JSON ({json_err}): {raw_suggestions_output}")
                 return None # Avoid returning potentially wrong extractions
        except Exception as e:
             logger.error(f"Suggestion generation error: {e}", exc_info=True)
             return None


    # --- UPDATED query method ---
    # ADD conversation_history parameter
    def query(self, query_text: str, conversation_history: Optional[List[ChatMessage]] = None) -> RAGResponse:
        """
        Executes RAG pipeline: Routing -> Score Check -> Generation + Sufficiency Check -> Fallback -> Suggestions.
        Uses generator's flag for supplemental fallback decision and passes history to generator.
        """
        logger.info(f"Processing query (sync): '{query_text[:50]}...' " + (f"with {len(conversation_history)} history messages" if conversation_history else "without history"))

        if not self.generator.llm: # Check main LLM
            logger.critical("Main Generator LLM unavailable.")
            return RAGResponse(answer="Sorry, the main language model is unavailable.", sources=[], suggestions=None)

        needs_direct_fallback = False
        needs_supplemental_fallback = False
        decision = 'NONE'
        target_metadata = None
        retrieved_docs = []
        high_score_docs = []
        initial_answer = ""
        final_answer = ""
        sources_for_response = []
        suggestions = None

        # --- 1. Route Query ---
        try:
            decision, target_metadata = self._route_query_llm(query_text)
            if decision == 'NONE':
                logger.warning("LLM router decided NONE. Triggering direct fallback.")
                needs_direct_fallback = True
        except Exception as route_err:
            logger.error(f"Fatal routing error: {route_err}", exc_info=True)
            needs_direct_fallback = True # Treat routing error as needing fallback

        # --- 2. Execute Based on Route ---
        if not needs_direct_fallback:
            if decision.startswith('SQL:'):
                # --- SQL Logic (No change needed for history here) ---
                parts = decision.split(':', 2); db_key = parts[1]; target_identifier = parts[2]
                target_db_utility = self.uploads_db_utility if db_key == 'UPLOADS' else self.rag_db_utility
                if target_db_utility and target_metadata:
                    db_name = getattr(target_db_utility._engine.url, 'database', 'Unknown DB'); table_name = target_metadata.get("target_table_name", target_identifier)
                    logger.info(f"Executing SQL route for: {db_name}/{table_name}")
                    try:
                        raw_sql = generate_sql_query(query_text, target_db_utility, target_metadata) # Assumes this function exists
                        if not raw_sql: raise ValueError("SQL generation failed.")
                        cleaned_sql = clean_sql_string(raw_sql) # Assumes this function exists
                        if not cleaned_sql: raise ValueError("Generated SQL invalid.")
                        db_execution_result = target_db_utility.run(cleaned_sql)
                        final_answer = f"Result from database table '{table_name}':\n```\n{db_execution_result}\n```"
                        # Format SQL sources clearly
                        sources_for_response = [{
                            "source": f"Database: {db_name}",
                            "type": "sql",
                            "details": f"Executed query on table '{table_name}'",
                            "query": cleaned_sql,
                            "content_snippet": str(db_execution_result)[:200]+"..."
                        }]
                    except Exception as e:
                        logger.error(f"SQL processing failed: {e}", exc_info=True)
                        final_answer = f"Sorry, I encountered an error trying to query the structured data: {e}"
                        sources_for_response = [] # No successful source
                else:
                     logger.error(f"SQL route failed: DB utility for key '{db_key}' or target metadata missing.")
                     final_answer = "Configuration error: Cannot access the required database for this query."
                     sources_for_response = []

            elif decision == 'VECTOR_STORE':
                # --- Vector Store Logic ---
                logger.info("LLM routed to Vector Store. Retrieving and checking scores...")
                RELEVANCE_THRESHOLD = 0.8 # Keep threshold as defined in original

                try:
                    # 2a. Retrieve Documents using EnhancedRetriever
                    # Assuming final_retriever is the one doing score-based retrieval
                    retrieved_docs = self.retriever_wrapper.final_retriever.invoke(query_text)

                    if not retrieved_docs:
                        logger.warning("No documents found by vector retriever. Triggering direct fallback.")
                        needs_direct_fallback = True
                    else:
                        # 2b. Filter by Score Threshold
                        high_score_docs = []
                        for doc in retrieved_docs:
                             score = doc.metadata.get('relevance_score')
                             if score is not None:
                                 try:
                                     if float(score) >= RELEVANCE_THRESHOLD:
                                         high_score_docs.append(doc)
                                     else:
                                          logger.debug(f"Doc '{doc.metadata.get('identifier', 'unknown')}' score ({score}) below threshold {RELEVANCE_THRESHOLD}.")
                                 except (ValueError, TypeError):
                                     logger.warning(f"Could not parse relevance score '{score}' for doc '{doc.metadata.get('identifier', 'unknown')}'. Skipping threshold check for this doc.")
                                     # Decide whether to include docs with unparsable scores. Let's exclude them.
                             else:
                                 logger.debug(f"Doc '{doc.metadata.get('identifier', 'unknown')}' missing relevance score. Excluding from high score list.")


                        if not high_score_docs:
                             logger.warning(f"No documents met relevance threshold ({RELEVANCE_THRESHOLD}). Triggering direct fallback.")
                             needs_direct_fallback = True
                        else:
                             logger.info(f"{len(high_score_docs)} documents met relevance threshold. Proceeding to generation.")
                             # --- 3. Generate Initial Answer & Check Sufficiency ---
                             try:
                                 # --- PASS HISTORY TO GENERATOR ---
                                 # This is the key change: pass conversation_history here
                                 initial_answer, is_sufficient = self.generator.generate_answer(
                                     query=query_text,
                                     retrieved_docs=high_score_docs, # Pass relevant docs
                                     conversation_history=conversation_history # Pass history
                                 )
                                 # ---------------------------------

                                 if not is_sufficient:
                                     logger.info("Generator indicated initial answer is insufficient. Triggering supplemental fallback.")
                                     needs_supplemental_fallback = True
                                     # Keep initial_answer for the first part
                                     # Keep high_score_docs for source listing
                                 else:
                                     logger.info("Generator indicated initial answer is sufficient.")
                                     final_answer = initial_answer # Use the grounded answer
                                     sources_for_response = RAGResponse.format_sources(high_score_docs)

                             except Exception as e:
                                 logger.error(f"Grounded generation or sufficiency check failed: {e}", exc_info=True)
                                 initial_answer = f"(Error during answer generation from context: {e})"
                                 # Proceed to supplemental fallback even if initial gen fails
                                 needs_supplemental_fallback = True
                                 # We have no initial answer, but might still provide sources
                                 sources_for_response = RAGResponse.format_sources(high_score_docs)


                except Exception as e:
                    logger.error(f"Vector retrieval or score check failed: {e}", exc_info=True)
                    needs_direct_fallback = True # Fallback if retrieval step fails

        # --- 4. Execute Fallback (Direct or Supplemental) ---
        # This logic remains the same as original, using FALLBACK_PROMPT
# --- 4. Execute Fallback (Direct or Supplemental) ---
        if needs_direct_fallback:
            logger.info(f"Executing direct fallback for query '{query_text[:50]}...'")
            if not self.generator.llm:
                logger.error("Fallback failed: Main LLM unavailable.")
                final_answer = "Sorry, I couldn't find relevant documents and the language model is unavailable to answer from general knowledge."
                sources_for_response = []
            else:
                try:
                    # --- MODIFICATION START ---
                    if conversation_history:
                        # Use fallback prompt WITH history
                        logger.debug("Using fallback prompt with history.")
                        formatted_history = format_chat_history(conversation_history)
                        fallback_input = {"question": query_text, "chat_history": formatted_history}
                        # Ensure FALLBACK_PROMPT_WITH_HISTORY is imported from prompts.py
                        fallback_chain = FALLBACK_PROMPT_WITH_HISTORY | self.generator.llm | StrOutputParser()
                        fallback_answer = fallback_chain.invoke(fallback_input)
                        # Adjust prefix to indicate history was considered
                        final_answer = f"Based on our conversation history and general knowledge:\n{fallback_answer}"
                        # Source indicates history was primary attempt
                        sources_for_response = [{"source": "internal-knowledge", "type": "internal", "content_snippet": "Answer generated using conversation history and/or language model's internal knowledge."}]
                    else:
                        # Use original fallback prompt WITHOUT history
                        logger.debug("Using fallback prompt without history.")
                        fallback_input = {"question": query_text}
                        # Ensure FALLBACK_PROMPT is imported from prompts.py
                        fallback_chain = FALLBACK_PROMPT | self.generator.llm | StrOutputParser()
                        fallback_answer = fallback_chain.invoke(fallback_input)
                        final_answer = f"I couldn't find specific documents relevant to your query.\n\nBased on general knowledge:\n{fallback_answer}"
                        sources_for_response = [{"source": "LLM Internal Knowledge", "type": "internal", "content_snippet": "Answer generated using the language model's internal knowledge."}]
                    # --- MODIFICATION END ---

                    logger.info("Direct fallback answer generated successfully.")
                except Exception as fallback_err:
                    logger.error(f"Direct fallback generation failed: {fallback_err}", exc_info=True)
                    final_answer = "Sorry, I encountered an error while trying to answer from general knowledge."
                    sources_for_response = []

        elif needs_supplemental_fallback:
            logger.info(f"Executing supplemental fallback for query '{query_text[:50]}...'")
            if not self.generator.llm:
                 logger.error("Fallback supplement failed: Main LLM unavailable.")
                 # Use initial answer if available, otherwise state error
                 final_answer = (initial_answer if initial_answer else "(Could not generate initial answer from context.)") + "\n\nAdditionally, the language model is unavailable to supplement this answer."
                 # Keep sources from high-score docs if available
                 sources_for_response = RAGResponse.format_sources(high_score_docs) if high_score_docs else []
            else:
                try:
                    # --- MODIFICATION START for supplement ---
                    if conversation_history:
                        # Use fallback prompt WITH history for the supplement
                        logger.debug("Using fallback prompt with history for supplement.")
                        formatted_history = format_chat_history(conversation_history)
                        fallback_input = {"question": query_text, "chat_history": formatted_history}
                        fallback_chain = FALLBACK_PROMPT_WITH_HISTORY | self.generator.llm | StrOutputParser()
                    else:
                        # Use original fallback prompt WITHOUT history for the supplement
                        logger.debug("Using fallback prompt without history for supplement.")
                        fallback_input = {"question": query_text}
                        fallback_chain = FALLBACK_PROMPT | self.generator.llm | StrOutputParser()

                    fallback_supplement = fallback_chain.invoke(fallback_input)

                    # Construct the two-part answer carefully
                    transition_phrase = "\n\nAs the provided documents may not contain all details, supplementing with the model's internal knowledge:\n\n"
                    final_answer = (initial_answer if initial_answer else "(Could not generate initial answer from context.)") + transition_phrase + fallback_supplement
                    # Combine sources: High-score docs + Internal marker
                    formatted_doc_sources = RAGResponse.format_sources(high_score_docs)
                    internal_source = {"source": "LLM Internal Knowledge", "type": "internal", "content_snippet": "Answer supplemented using the language model's internal knowledge."}
                    sources_for_response = formatted_doc_sources + [internal_source]
                    logger.info("Fallback supplement generated successfully.")
                except Exception as fallback_err:
                    logger.error(f"Fallback supplement generation failed: {fallback_err}", exc_info=True)
                    final_answer = (initial_answer if initial_answer else "(Could not generate initial answer from context.)") + "\n\nSorry, there was an error supplementing this answer."
                    sources_for_response = RAGResponse.format_sources(high_score_docs) if high_score_docs else []


        # --- 5. Generate Suggestions ---
        # This logic remains the same
        generate_sugg = bool(final_answer and not final_answer.startswith("Sorry") and not final_answer.startswith("Error") and not final_answer.startswith("Configuration error"))
        if generate_sugg and self.light_llm:
             try:
                 suggestions = self._generate_suggestions(query_text, final_answer)
             except Exception as sugg_err:
                 logger.error(f"Suggestion generation failed: {sugg_err}", exc_info=True) # Log error but continue
                 suggestions = None
        else:
            suggestions = None # Ensure it's None if not generated


        # --- 6. Construct Final Response Object ---
        # Ensure final_answer is always a string
        if not isinstance(final_answer, str) or not final_answer:
             final_answer = "Sorry, I could not generate a valid response for your query."
             logger.error("Final answer was empty or not a string before returning.")
             # Reset sources if the answer generation fundamentally failed
             # sources_for_response = [] # Decide if sources should be cleared

        logger.info(f"Query processing complete (sync): '{query_text[:50]}...'")
        final_response_obj = RAGResponse(
            answer=final_answer.strip(), # Trim whitespace
            sources=sources_for_response,
            suggestions=suggestions
        )

        return final_response_obj