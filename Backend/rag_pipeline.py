# RAG_Project/MY_RAG/Backend/rag_pipeline.py

import logging
import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

# --- Pydantic and Langchain Core Imports ---
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# --- Local Module Imports ---
try:
    from config import settings
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
    from prompts import (
        create_rag_prompt_template, format_chat_history,
        FALLBACK_PROMPT, FALLBACK_PROMPT_WITH_HISTORY,
        ROUTER_PROMPT, PYTHON_GENERATION_PROMPT # Ensure PYTHON_GENERATION_PROMPT is the latest version
    )
    from data_science_executor import DataScienceExecutor, DataScienceExecutorError
    from models import ChatMessage
except ImportError as e:
     logging.exception(f"Error importing local modules in rag_pipeline.py: {e}", exc_info=True)
     raise

logger = logging.getLogger(__name__)

# --- RAG Response Model (Unchanged) ---
class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: Optional[List[str]] = None

    @staticmethod
    def format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
        sources_data = []
        if not docs: return sources_data
        seen_identifiers = set()
        for doc in docs:
            identifier = doc.metadata.get('identifier', doc.metadata.get('source'))
            if not identifier:
                content_hash = hash(doc.page_content[:50])
                identifier = f"unknown_{content_hash}_{len(seen_identifiers)}"
            if identifier in seen_identifiers: continue
            seen_identifiers.add(identifier)
            source_label = doc.metadata.get('file_name', identifier)
            source_type_flag = doc.metadata.get('type', 'vector')
            source_info = {
                "source": source_label, "type": source_type_flag,
                "content_snippet": doc.page_content[:200] + "..." if source_type_flag != 'plot_png_base64' else "Generated Plot",
                "relevance_score": doc.metadata.get('relevance_score'),
                "identifier": identifier, "page": doc.metadata.get('page'),
                "summary_id": doc.metadata.get('summary_id'),
                "uploaded_by": doc.metadata.get('uploaded_by'),
                "data": doc.metadata.get('data') if source_type_flag == 'plot_png_base64' else None
            }
            sources_data.append({k: v for k, v in source_info.items() if v is not None})
        return sources_data

# --- RAG Pipeline Class ---
class RAGPipeline:
    def __init__(self, indexer: Indexer):
        self.retriever_wrapper = EnhancedRetriever(indexer)
        self.generator = AnswerGenerator()
        self.rag_db_utility: Optional[SQLDatabase] = get_rag_db_utility()
        self.uploads_db_utility: Optional[SQLDatabase] = get_uploads_db_utility()
        self.light_llm = None
        try:
            if settings.light_llm_model_name:
                 self.light_llm = ChatGoogleGenerativeAI(model=settings.light_llm_model_name, temperature=0.1)
                 logger.info(f"Pipeline Light LLM (Routing/Suggestions) initialized: {settings.light_llm_model_name}")
            else:
                 logger.warning("LIGHT_LLM_MODEL_NAME not set, routing/suggestion features may be limited.")
        except Exception as e:
            logger.error(f"Failed to init Pipeline Light LLM: {e}", exc_info=True)
        self.ds_executor: Optional[DataScienceExecutor] = None
        try:
            self.ds_executor = DataScienceExecutor()
        except Exception as e:
             logger.error(f"Failed to initialize DataScienceExecutor: {e}", exc_info=True)
        logger.info("RAG Pipeline initialized.")
        # (Availability logging...)

    def _format_metadata_for_router(self, docs: List[Document]) -> str:
        # (Unchanged)
        if not docs: return "No potentially relevant data sources found."
        formatted_sources = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            source_id = meta.get('identifier', f'unknown_{i}')
            data_type = meta.get('data_type', 'unknown')
            source_type = meta.get('source_type', 'unknown')
            doc_summary = meta.get('summary', doc.page_content[:300])
            source_info = f"Source {i+1}:\n  ID: {source_id}\n  Type: {data_type} ({source_type})\n  Summary: {doc_summary}\n"
            if data_type == 'structured' and 'columns' in meta and isinstance(meta['columns'], list):
                 col_names = [str(col.get('column', '?')) for col in meta['columns'][:15]]
                 source_info += f"  Columns: {', '.join(col_names)}{'...' if len(meta['columns']) > 15 else ''}\n"
            elif data_type == 'unstructured' and 'keywords' in meta and isinstance(meta['keywords'], list):
                 source_info += f"  Keywords: {', '.join(meta['keywords'][:10])}\n"
            formatted_sources.append(source_info.strip())
        return "\n---\n".join(formatted_sources)

    def _route_query_llm(self, query_text: str) -> tuple[str, Optional[Dict[str, Any]]]:
        # (Unchanged from previous correct version)
        logger.info(f"Routing query (sync): '{query_text[:50]}...'")
        decision_type = 'VECTOR_STORE'
        target_metadata = None
        if not self.light_llm:
            logger.error("Routing LLM unavailable. Falling back to VECTOR_STORE.")
            return 'VECTOR_STORE', None
        try:
            base_retriever: BaseRetriever = self.retriever_wrapper.base_retriever
            metadata_retriever_k = getattr(settings, 'metadata_retriever_k', 7)
            # Ensure the base retriever's vectorstore is valid before creating the metadata retriever
            if not hasattr(base_retriever, 'vectorstore') or not base_retriever.vectorstore:
                 logger.error("Base vectorstore unavailable for metadata routing. Falling back to VECTOR_STORE.")
                 return 'VECTOR_STORE', None
            metadata_retriever = base_retriever.vectorstore.as_retriever(search_kwargs={'k': metadata_retriever_k})
            retrieved_metadata_docs = metadata_retriever.invoke(query_text)

            if not retrieved_metadata_docs:
                logger.warning("No metadata docs found during initial routing retrieval. Defaulting to VECTOR_STORE.")
                return 'VECTOR_STORE', None

            formatted_sources_str = self._format_metadata_for_router(retrieved_metadata_docs)
            router_input = {"available_sources": formatted_sources_str, "query_text": query_text}
            router_chain = ROUTER_PROMPT | self.light_llm | StrOutputParser()
            raw_decision = router_chain.invoke(router_input)
            decision = raw_decision.strip().upper()
            logger.info(f"LLM Router Decision (sync, raw): {raw_decision}, Parsed: {decision}") # Log raw and parsed

            # Find the chosen document based on the decision ID, BEFORE refining decision type
            chosen_doc = None
            target_identifier_from_llm = None
            if ':' in decision:
                 try:
                     decision_prefix, target_identifier_from_llm = decision.split(':', 1)
                     target_identifier_from_llm = target_identifier_from_llm.strip()
                     # Find the corresponding doc from the initially retrieved metadata docs
                     chosen_doc = next((doc for doc in retrieved_metadata_docs if doc.metadata.get('identifier', '').lower() == target_identifier_from_llm.lower()), None)
                     if chosen_doc:
                         target_metadata = chosen_doc.metadata
                         logger.info(f"Routing identified target source: '{target_metadata.get('identifier')}' based on LLM decision.")
                     else:
                         logger.warning(f"LLM chose target '{target_identifier_from_llm}' but it wasn't found in retrieved metadata docs.")
                         # Decide fallback - maybe VECTOR_STORE is safer if specific target fails?
                         target_metadata = None # Reset target metadata if chosen doc not found
                 except ValueError:
                     logger.warning(f"Could not parse target ID from LLM decision: {decision}")
                     target_metadata = None # Reset if parsing fails


            # Determine final decision_type based on parsed decision and validity of chosen_doc/target_metadata
            if decision == 'VECTOR_STORE':
                decision_type = 'VECTOR_STORE'
            elif decision == 'NONE':
                decision_type = 'NONE'
            elif decision.startswith('SQL:'):
                if chosen_doc and target_metadata: # Ensure we found the target
                    target_identifier = target_metadata.get('identifier')
                    db_key = 'NONE'
                    target_db_name = target_metadata.get('target_database')
                    if target_db_name == 'RAG_DB_UPLOADS' and self.uploads_db_utility: db_key = 'UPLOADS'
                    elif target_db_name == 'RAG_DB' and self.rag_db_utility: db_key = 'MAIN'

                    if db_key != 'NONE':
                        # Ensure table name exists in metadata
                        if 'target_table_name' not in target_metadata or not target_metadata['target_table_name']:
                             target_metadata['target_table_name'] = target_identifier # Fallback if missing
                             logger.warning(f"Target table name missing for SQL route {target_identifier}, using identifier as fallback.")

                        decision_type = f"SQL:{db_key}:{target_identifier}" # Use the validated identifier
                    else:
                        logger.warning(f"SQL route chosen for {target_identifier}, but required DB utility ('{target_db_name}') is unavailable.")
                        decision_type = 'VECTOR_STORE' # Fallback if DB is unavailable
                        target_metadata = None # Clear metadata if falling back
                else:
                    logger.warning(f"SQL route chosen, but target metadata '{target_identifier_from_llm}' not found or invalid.")
                    decision_type = 'VECTOR_STORE' # Fallback if target doc invalid
                    target_metadata = None
            elif decision.startswith('PYTHON_ANALYSIS:') or decision == 'PYTHON_ANALYSIS':
                 # If specific target was identified and found:
                 if chosen_doc and target_metadata:
                      decision_type = 'PYTHON_ANALYSIS'
                      # Metadata already set from chosen_doc
                 # If 'PYTHON_ANALYSIS' without specific target or target not found:
                 else:
                      decision_type = 'PYTHON_ANALYSIS'
                      target_metadata = None # No specific source metadata
                      logger.info("Python Analysis route chosen without specific target or target not found in metadata.")
            else: # Default fallback for unexpected decisions
                 logger.warning(f"Unknown router decision '{decision}'. Defaulting to VECTOR_STORE.")
                 decision_type = 'VECTOR_STORE'
                 target_metadata = None

        except Exception as e:
             logger.error(f"LLM routing step failed: {e}", exc_info=True)
             decision_type = 'VECTOR_STORE'
             target_metadata = None # Ensure metadata is cleared on error

        logger.info(f"Final routing decision: {decision_type}")
        if target_metadata: logger.debug(f"Target metadata associated with decision: ID='{target_metadata.get('identifier')}', Type='{target_metadata.get('data_type')}'")
        else: logger.debug("No specific target metadata associated with decision.")
        return decision_type, target_metadata

    def _generate_suggestions(self, user_query: str, ai_answer: str) -> Optional[List[str]]:
        # (Unchanged)
        if not user_query or not ai_answer or not self.light_llm: return None
        logger.info("Generating suggestions (sync)...")
        try:
            suggestion_prompt_template = """
Given the user's query and the AI's answer, suggest 3 concise and relevant follow-up questions or actions.
Phrase them as if the user is asking.
Respond ONLY with a valid JSON list of strings, like ["Suggestion 1", "Suggestion 2", "Suggestion 3"].

User Query: "{user_query}"
AI Answer: "{ai_answer}"

Suggested follow-up questions/actions (JSON list):
            """
            suggestion_prompt = PromptTemplate.from_template(suggestion_prompt_template)
            suggestion_chain = suggestion_prompt | self.light_llm | StrOutputParser()
            raw_suggestions_output = suggestion_chain.invoke({"user_query": user_query, "ai_answer": ai_answer})
            match = re.search(r"\[.*\]", raw_suggestions_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    suggestions = json.loads(json_str)
                    if isinstance(suggestions, list) and all(isinstance(s, str) for s in suggestions): return suggestions[:3]
                except json.JSONDecodeError: pass # Fallthrough if JSON parsing fails
            extracted = re.findall(r'"([^"]+)"', raw_suggestions_output);
            if extracted: return extracted[:3]
            return None
        except Exception as e:
             logger.error(f"Suggestion generation error: {e}", exc_info=True)
             return None

    def _clean_python_code(self, raw_code: str) -> str:
        # (Unchanged)
        logger.debug("Cleaning generated Python code...")
        cleaned = re.sub(r"^\s*```python\s*", "", raw_code, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()
        logger.debug(f"Cleaned code snippet: {cleaned[:200]}...")
        return cleaned

    def query(self, query_text: str, conversation_history: Optional[List[ChatMessage]] = None) -> RAGResponse:
        # (Initialization unchanged)
        logger.info(f"Processing query (sync): '{query_text[:50]}...' " + (f"with {len(conversation_history)} history messages" if conversation_history else "without history"))
        if not self.generator.llm:
            return RAGResponse(answer="Sorry, the main language model is unavailable.", sources=[], suggestions=None)

        needs_direct_fallback = False; needs_supplemental_fallback = False
        decision = 'NONE'; target_metadata = None; retrieved_docs = []; high_score_docs = []
        initial_answer = ""; final_answer = ""; sources_for_response = []; suggestions = None
        intermediate_data: Optional[Any] = None

        # --- Routing ---
        try:
            decision, target_metadata = self._route_query_llm(query_text)
            if decision == 'NONE': needs_direct_fallback = True
        except Exception as route_err:
            logger.error(f"Fatal routing error: {route_err}", exc_info=True); needs_direct_fallback = True

        # --- Execute Based on Route ---
        if not needs_direct_fallback:
            try:
                # --- SQL Execution (Unchanged) ---
                if decision.startswith('SQL:'):
                    parts = decision.split(':', 2); db_key = parts[1]; target_identifier = parts[2]
                    target_db_utility = self.uploads_db_utility if db_key == 'UPLOADS' else self.rag_db_utility
                    # Ensure target_metadata exists for SQL execution
                    if target_db_utility and target_metadata and target_metadata.get('identifier') == target_identifier:
                        db_name = getattr(target_db_utility._engine.url, 'database', 'Unknown DB')
                        table_name = target_metadata.get("target_table_name", target_identifier) # Assumes table name is in metadata now
                        logger.info(f"Executing SQL route for: {db_name}/{table_name}")
                        try:
                            raw_sql = generate_sql_query(query_text, target_db_utility, target_metadata)
                            if not raw_sql: raise ValueError("SQL generation failed.")
                            cleaned_sql = clean_sql_string(raw_sql)
                            if not cleaned_sql: raise ValueError("Generated SQL invalid.")
                            db_execution_result = target_db_utility.run(cleaned_sql)
                            intermediate_data = db_execution_result
                            final_answer = f"Result from database table '{table_name}':\n```\n{db_execution_result}\n```"
                            # Include metadata source info
                            sql_source = {"source": f"{target_metadata.get('original_filename', target_identifier)} (DB Table: {table_name})", "type": "sql", "details": f"Executed query", "query": cleaned_sql, "content_snippet": str(db_execution_result)[:200]+"...", "identifier": target_identifier}
                            if 'summary_id' in target_metadata: sql_source['summary_id'] = target_metadata['summary_id']
                            sources_for_response = [sql_source]
                        except Exception as e:
                            logger.error(f"SQL execution failed for table '{table_name}': {e}", exc_info=True)
                            final_answer = f"Sorry, I encountered an error trying to query the structured data '{target_identifier}': {e}"; sources_for_response = []
                    else:
                         logger.error(f"SQL route configuration error: DB utility or metadata mismatch for target '{target_identifier}'.")
                         final_answer = "Configuration error: Could not access the required database or metadata for this SQL query."; sources_for_response = []

                # --- Vector Store RAG (Unchanged) ---
                elif decision == 'VECTOR_STORE':
                    logger.info("LLM routed to Vector Store. Retrieving and checking scores...")
                    RELEVANCE_THRESHOLD = getattr(settings, 'relevance_threshold', 0.75)
                    retrieved_docs = self.retriever_wrapper.retrieve_documents(query_text)
                    if not retrieved_docs:
                        logger.warning("Vector store retrieval returned 0 documents.")
                        needs_direct_fallback = True
                    else:
                        high_score_docs = [doc for doc in retrieved_docs if float(doc.metadata.get('relevance_score', 0.0)) >= RELEVANCE_THRESHOLD]
                        if not high_score_docs:
                             logger.warning(f"Vector store retrieval returned {len(retrieved_docs)} docs, but none met threshold {RELEVANCE_THRESHOLD}.")
                             needs_direct_fallback = True # Treat low relevance as needing fallback
                        else:
                             logger.info(f"Retrieved {len(high_score_docs)} relevant documents from vector store.")
                             initial_answer, is_sufficient = self.generator.generate_answer(query=query_text, retrieved_docs=high_score_docs, conversation_history=conversation_history)
                             sources_for_response = RAGResponse.format_sources(high_score_docs)
                             if not is_sufficient:
                                 logger.info("Initial answer deemed insufficient by LLM check.")
                                 needs_supplemental_fallback = True
                             else:
                                 final_answer = initial_answer

                # --- Python Analysis Execution (Incorporating Metadata into Prompt) ---
                elif decision == 'PYTHON_ANALYSIS':
                    logger.info("LLM routed to Python Analysis.")
                    if not self.ds_executor:
                         final_answer = "Sorry, the Python analysis service is currently unavailable."
                         sources_for_response = [{"source": "System", "type": "error", "content_snippet": final_answer}]
                    else:
                        # Initialize defaults
                        data_info = "No specific data source identified. Proceeding with general Python execution based on the query."
                        initialization_code = "# No specific data loaded. Write general Python code based on the query."
                        user_code_instructions = "# Write Python code to answer the query based on general knowledge or common libraries."
                        db_password_to_hide = "" # Initialize password variable

                        # --- Populate context if target_metadata (structured source) exists ---
                        if target_metadata: # Check if metadata exists first
                            source_id = target_metadata.get('identifier', 'Unknown Source')
                            # *** FIX for data_type warning: Ensure data_type is fetched if target_metadata exists ***
                            current_data_type = target_metadata.get('data_type', 'unknown') # Use a different name initially if needed
                            logger.info(f"Python route target identified: '{source_id}' (Type: {current_data_type})")

                            if current_data_type == 'structured':
                                 table_name = target_metadata.get('target_table_name')
                                 db_name_meta = target_metadata.get('target_database')

                                 if table_name and db_name_meta:
                                     # --- Determine correct DB URL for executor ---
                                     base_db_url_str = settings.postgres_uploads_url if db_name_meta == 'RAG_DB_UPLOADS' else settings.postgres_url
                                     try:
                                         parsed_url = urlparse(base_db_url_str)
                                         db_user = parsed_url.username or 'postgres'
                                         db_password_to_hide = parsed_url.password or 'password' # Store for later use
                                         db_host_for_executor = 'host.docker.internal'
                                         db_port_for_executor = parsed_url.port or 5432
                                         db_database = parsed_url.path.lstrip('/') if parsed_url.path else db_name_meta
                                         if not db_database: raise ValueError(f"Could not determine database name from URL: {base_db_url_str} or metadata: {db_name_meta}")

                                         correct_db_url_for_executor = f"postgresql+psycopg://{db_user}:{db_password_to_hide}@{db_host_for_executor}:{db_port_for_executor}/{db_database}"
                                         logger.info(f"Constructed DB URL for executor (connecting to host {db_host_for_executor}:{db_port_for_executor}): {correct_db_url_for_executor.replace(db_password_to_hide, '****')}")

                                     except Exception as parse_err:
                                         logger.error(f"Error constructing executor DB URL: {parse_err}", exc_info=True)
                                         # Fallback to general Python if DB connection fails
                                         target_metadata = None # Clear metadata if we can't connect
                                         data_info = "Error preparing database connection. Proceeding with general Python execution."
                                         initialization_code = "# Database connection failed to be configured."
                                         user_code_instructions = "# Write Python code based on query and general knowledge; database access failed."
                                         # Continue to code generation below with these modified inputs

                                     # --- If DB URL constructed successfully, Format Metadata for Prompt ---
                                     if target_metadata: # Check if still valid after potential error above
                                         data_info_parts = [
                                             f"Target data source: Table `{table_name}` from Database `{db_database}`.", # Use backticks for code style
                                             f"Source File: {target_metadata.get('original_filename', source_id)}",
                                             f"Description: {target_metadata.get('dataset_description', 'N/A')}",
                                             f"Row Count: {target_metadata.get('row_count', 'N/A')}",
                                             "Columns (Name, Type, Description):"
                                         ]
                                         columns_metadata = target_metadata.get('columns', [])
                                         if columns_metadata:
                                             for col in columns_metadata:
                                                 # Clean potential NUL bytes from metadata strings before adding to prompt
                                                 col_name = str(col.get('column', 'N/A')).replace('\x00', '')
                                                 col_dtype = str(col.get('dtype', 'N/A')).replace('\x00', '')
                                                 col_desc = str(col.get('description', 'N/A')).replace('\x00', '')
                                                 col_info = f"  - `{col_name}` ({col_dtype}): {col_desc}" # Use backticks
                                                 data_info_parts.append(col_info)
                                         else:
                                             data_info_parts.append("  (Column details not available in metadata)")
                                         data_info = "\n".join(data_info_parts)

                                         # --- Initialization code to load DF and print info ---
                                         initialization_code = f"""
# --- ADD THIS LINE ---
print("--- EXECUTOR: STARTING INITIALIZATION CODE ---")
# --- END ADD ---

import pandas as pd
from sqlalchemy import create_engine, text

# DB URL connecting from executor container to host DB ({db_host_for_executor}:{db_port_for_executor})
DB_URL = "{correct_db_url_for_executor}"
df = None # Define df initially
try:
    print("--- EXECUTOR: Attempting DB connection and load ---") # Add another print
    engine = create_engine(DB_URL)
    with engine.connect() as connection:
        query = text(f'SELECT * FROM "{table_name}"') # Assuming standard SQL quoting
        df = pd.read_sql(query, connection)
    print(f"--- EXECUTOR: Successfully loaded data from table '{table_name}'. Shape: {{df.shape if df is not None else 'Load Failed'}} ---") # Add markers

    # Print dataframe info immediately after loading for the LLM context
    if df is not None and not df.empty:
        print("\\n--- DataFrame Info ---") # Use double backslash for newline in f-string
        print(f"Columns: {{df.columns.tolist()}}") # Use f-string for columns
        print("Head:")
        print(df.head().to_markdown(index=False)) # Print head() for context
        print("----------------------\\n") # Use double backslash
    elif df is not None and df.empty:
        print("--- EXECUTOR: DataFrame loaded successfully but is empty. ---")

except Exception as e:
    # Log the error with the connection string (password obscured)
    print(f"--- EXECUTOR: ERROR loading data from table '{table_name}' using URL '{correct_db_url_for_executor.replace(db_password_to_hide, '****')}': {{e}} ---") # Add markers
if df is None:
    print("--- EXECUTOR: DataFrame 'df' is None after load attempt. ---") # Add marker
# --- END OF initialization_code ---
"""
                                         user_code_instructions = f"# Analyze the DataFrame 'df' (schema above, head printed in output) to answer the query."
                                 # End if target_metadata still valid after potential DB URL error

                                 else: # Handle case where structured data identified but no table/db info found in metadata
                                     data_info = f"Target data source: '{source_id}' (Type: structured). Missing table/database details in metadata."
                                     initialization_code = "# No DataFrame loaded as database/table details were missing."
                                     user_code_instructions = "# Analyze based on the query and general knowledge; specific structured data could not be loaded."

                            else: # Handle case where target_metadata exists but is not structured
                                 # *** FIX for data_type warning: Use current_data_type here ***
                                 data_info = f"Target data source: '{source_id}' (Type: {current_data_type}). Cannot load non-structured data into DataFrame automatically."
                                 initialization_code = "# No DataFrame loaded as target was not identified as structured data."
                                 user_code_instructions = "# Analyze based on the query and general knowledge, no DataFrame available."
                        # End if target_metadata exists

                        # --- Handle general Python analysis (no specific target_metadata) ---
                        # Defaults already set above, no changes needed here


                        # --- Generate and Execute Code ---
                        formatted_history = format_chat_history(conversation_history)
                        code_gen_llm = self.generator.llm
                        if not code_gen_llm: raise ValueError("LLM for code generation unavailable.")

                        # Make sure PYTHON_GENERATION_PROMPT is imported correctly
                        if not PYTHON_GENERATION_PROMPT:
                             logger.error("PYTHON_GENERATION_PROMPT not available from prompts module.")
                             raise ValueError("Python code generation prompt is missing.")

                        code_gen_chain = PYTHON_GENERATION_PROMPT | code_gen_llm | StrOutputParser()
                        code_gen_input = {
                             "question": query_text, "data_info": data_info,
                             "chat_history": formatted_history,
                             "initialization_code": initialization_code,
                             "user_code_instructions": user_code_instructions
                        }
                        logger.debug(f"Code Generation Input - data_info:\n{data_info}") # Log data_info passed to prompt
                        raw_python_code = code_gen_chain.invoke(code_gen_input)
                        generated_python_code = self._clean_python_code(raw_python_code)
                        logger.debug(f"Generated Python code for execution:\n{generated_python_code}") # Log generated code

                        if not generated_python_code or generated_python_code.strip() == "":
                            final_answer = "Sorry, I couldn't generate valid Python code for this request."
                            sources_for_response = [{"source": "Code Generation", "type": "error", "content_snippet": final_answer}]
                        else:
                            try:
                                logger.info("Executing generated Python code...")
                                execution_result = self.ds_executor.execute_analysis(generated_python_code)
                                exec_stdout = execution_result.get("stdout", ""); exec_stderr = execution_result.get("stderr", "")
                                plot_base64 = execution_result.get("plot_png_base64"); exec_success = execution_result.get("execution_successful", False)
                                client_error = execution_result.get("error")

                                if client_error: # Error communicating with executor service
                                    logger.error(f"Executor client error: {client_error}")
                                    final_answer = f"Error communicating with execution service: {exec_stderr or client_error}"
                                    sources_for_response = [{"source": "Executor Client", "type": "error", "content_snippet": final_answer}]
                                elif not exec_success: # Execution started but failed
                                    logger.warning(f"Python execution failed. Stderr:\n{exec_stderr}")
                                    final_answer = f"Python analysis encountered an error."
                                    # Include stdout first as it might contain useful partial results or printed info
                                    if exec_stdout: final_answer += f"\n\nOutput before error:\n```\n{exec_stdout}\n```"
                                    final_answer += f"\n\nError details:\n```\n{exec_stderr}\n```" # Add stderr
                                    # *** No df variable here, so linter warning for df is ignorable ***
                                    sources_for_response = [{"source": "Python Execution", "type": "error", "content_snippet": exec_stderr[:300]+"..."}]
                                else: # Execution successful
                                    logger.info("Python execution successful.")
                                    final_answer = f"Python analysis successful."
                                    if exec_stdout: final_answer += f"\n\nOutput:\n```\n{exec_stdout}\n```"
                                    # Add source info, linking back to the original data source if target_metadata was used
                                    analysis_source_info = {"source": "Python Analysis Results", "type": "code_execution", "content_snippet": exec_stdout[:200]+"..."}
                                    if target_metadata and target_metadata.get('identifier'):
                                        analysis_source_info['source'] = f"{target_metadata.get('original_filename', target_metadata.get('identifier'))} (via Python)"
                                        analysis_source_info['identifier'] = target_metadata.get('identifier')
                                        if 'summary_id' in target_metadata: analysis_source_info['summary_id'] = target_metadata['summary_id']
                                    sources_for_response = [analysis_source_info]

                                    if plot_base64:
                                        logger.info("Plot generated during Python execution.")
                                        final_answer += "\n\nA plot was generated."
                                        sources_for_response.append({"source": "Generated Plot", "type": "plot_png_base64", "identifier": "plot_png_output", "data": plot_base64})

                            except DataScienceExecutorError as dse_err: # Error *calling* the executor client
                                logger.error(f"DataScienceExecutorError: {dse_err}", exc_info=True)
                                final_answer = f"Critical error running analysis: {dse_err}"; sources_for_response = [{"source": "Executor Client", "type": "error", "content_snippet": str(dse_err)[:300]+"..."}]
                            except Exception as exec_e: # Other unexpected errors during the execution phase
                                # *** Line ~415: No df variable here, so linter warning for df is ignorable ***
                                logger.error(f"Unexpected error during Python execution block: {exec_e}", exc_info=True)
                                final_answer = f"Unexpected error during Python analysis execution: {exec_e}"; sources_for_response = [{"source": "System", "type": "error", "content_snippet": final_answer}]


            except Exception as e: # Catch errors during the main execution phase (SQL, VECTOR_STORE, PYTHON_ANALYSIS blocks)
                logger.error(f"Error during '{decision}' execution phase: {e}", exc_info=True); needs_direct_fallback = True
                final_answer = f"An internal error occurred while trying to process your request using the '{decision}' method."; sources_for_response = [] # Generic error for user

        # --- Fallback Logic (Unchanged) ---
        if needs_direct_fallback:
            logger.info(f"Executing direct fallback for query '{query_text[:50]}...'")
            if not self.generator.llm: final_answer = "Language model unavailable."; sources_for_response = []
            else:
                try:
                    fallback_prompt_to_use = FALLBACK_PROMPT; fallback_input = {"question": query_text}; history_context = "general knowledge"
                    if conversation_history:
                        formatted_history = format_chat_history(conversation_history); fallback_input["chat_history"] = formatted_history
                        fallback_prompt_to_use = FALLBACK_PROMPT_WITH_HISTORY; history_context = "conversation history and general knowledge"
                    fallback_chain = fallback_prompt_to_use | self.generator.llm | StrOutputParser()
                    fallback_answer = fallback_chain.invoke(fallback_input)
                    final_answer = f"I couldn't find specific documents or complete the analysis.\n\nBased on {history_context}:\n{fallback_answer}"
                    sources_for_response = [{"source": "LLM Internal Knowledge", "type": "internal", "content_snippet": f"Answer generated using {history_context}."}]
                except Exception as fallback_err:
                    logger.error(f"Error during direct fallback generation: {fallback_err}", exc_info=True)
                    final_answer = "Error answering from general knowledge."; sources_for_response = []
        elif needs_supplemental_fallback:
            logger.info(f"Executing supplemental fallback for query '{query_text[:50]}...'")
            if not self.generator.llm: final_answer = (initial_answer or "") + "\n\nLanguage model unavailable to supplement."
            else:
                try:
                    fallback_prompt_to_use = FALLBACK_PROMPT; fallback_input = {"question": query_text}
                    if conversation_history:
                        formatted_history = format_chat_history(conversation_history); fallback_input["chat_history"] = formatted_history
                        fallback_prompt_to_use = FALLBACK_PROMPT_WITH_HISTORY
                    fallback_chain = fallback_prompt_to_use | self.generator.llm | StrOutputParser()
                    fallback_supplement = fallback_chain.invoke(fallback_input)
                    final_answer = (initial_answer or "") + "\n\nSupplementing with internal knowledge:\n\n" + fallback_supplement
                    internal_source = {"source": "LLM Internal Knowledge", "type": "internal", "content_snippet": "Answer supplemented."}
                    if not isinstance(sources_for_response, list): sources_for_response = []
                    sources_for_response.append(internal_source)
                except Exception as fallback_err:
                    logger.error(f"Error during supplemental fallback generation: {fallback_err}", exc_info=True)
                    final_answer = (initial_answer or "") + "\n\nError supplementing answer."

        # --- Generate Suggestions & Final Response (Unchanged) ---
        generate_sugg = bool(final_answer and not final_answer.startswith("Sorry") and not final_answer.startswith("Error"))
        if generate_sugg and self.light_llm:
             try: suggestions = self._generate_suggestions(query_text, final_answer)
             except Exception as sugg_err:
                 logger.error(f"Suggestion generation failed: {sugg_err}")
                 suggestions = None
        else: suggestions = None
        if not isinstance(final_answer, str) or not final_answer: final_answer = "Could not generate a valid response."
        if not isinstance(sources_for_response, list): sources_for_response = []

        logger.info(f"Query processing complete (sync): '{query_text[:50]}...'")
        return RAGResponse(answer=final_answer.strip(), sources=sources_for_response, suggestions=suggestions)

    def __del__(self):
        if hasattr(self, 'ds_executor') and self.ds_executor:
            try: self.ds_executor.close()
            except Exception as e: logger.error(f"Error closing DataScienceExecutor session: {e}", exc_info=True)