# sql_processing.py
import logging
import re
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
# Import PromptTemplate directly
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# Import the SQL_PROMPTS dictionary
from langchain.chains.sql_database.prompt import SQL_PROMPTS

from config import settings

logger = logging.getLogger(__name__)

# --- Global DB instance and other functions (db, clean_sql_string, get_sql_database_utility, is_structured_query) remain the same ---
try:
    db = SQLDatabase.from_uri(settings.postgres_url, engine_args={"pool_size": 5, "max_overflow": 2})
    logger.info("SQLDatabase connection initialized for SQL processing.")
except Exception as e: db = None; logger.error(f"Failed to initialize SQLDatabase connection: {e}", exc_info=True)
def clean_sql_string(sql: str) -> str: # ... implementation from previous step ...
    if not isinstance(sql, str): logger.error(f"Invalid input to clean_sql_string: expected string, got {type(sql)}"); return ""
    logger.debug(f"Attempting to clean SQL (Original): '{sql}'"); cleaned_sql = sql
    pattern = r"```(?:[a-zA-Z0-9]*)?\s*(.*?)\s*```"; match = re.search(pattern, cleaned_sql, re.DOTALL)
    if match: cleaned_sql = match.group(1); logger.debug(f"Cleaned SQL (after regex extraction): '{cleaned_sql}'")
    else:
        logger.debug("Regex did not find standard markdown fences."); prefixes_to_remove = ["```sql", "```", "sql"]
        temp_stripped = cleaned_sql.strip(); removed_prefix = False
        for prefix in prefixes_to_remove:
            if temp_stripped.lower().startswith(prefix):
                 cleaned_sql = temp_stripped[len(prefix):]; logger.debug(f"Cleaned SQL (after removing prefix '{prefix}'): '{cleaned_sql}'"); removed_prefix = True; break
        if not removed_prefix: cleaned_sql = cleaned_sql.strip().strip('`'); logger.debug(f"Cleaned SQL (no prefix detected, using basic strip): '{cleaned_sql}'")
    cleaned_sql = cleaned_sql.strip().strip('`'); logger.info(f"Final Cleaned SQL: '{cleaned_sql}'"); return cleaned_sql
def get_sql_database_utility() -> SQLDatabase | None: return db
def is_structured_query(question: str) -> bool: # ... implementation from previous step ...
    global db;
    if not db: sql_words = ["select", "from", "where", "join", "count", "sum", "avg", "group by", "having", "limit", "order by", "table", "column", "database"]; q_lower = question.lower(); return any(word in q_lower for word in sql_words)
    try:
        all_tables_info = db.get_table_info(db.get_usable_table_names()); schema_keywords = set(); lines = all_tables_info.lower().split('\n'); current_table = None
        for line in lines:
            if line.startswith('table:'): current_table = line.split('table:')[1].split('(')[0].strip(); schema_keywords.add(current_table)
            elif current_table and ':' in line and not line.startswith(' '): col_name = line.split(':')[0].strip(); schema_keywords.add(col_name)
        sql_words = {"select", "from", "where", "join", "count", "sum", "avg", "group", "having", "limit", "order", "insert", "update", "delete"}; keywords = schema_keywords.union(sql_words)
        q_lower = question.lower(); found = any(kw in q_lower for kw in keywords if len(kw) > 2); return found
    except Exception as e: logger.error(f"Failed structured query check: {e}", exc_info=True); sql_words = ["select", "from", "where", "join", "count", "complete","sum", "avg", "group by", "having", "limit", "order by", "table", "column", "database"]; q_lower = question.lower(); return any(word in q_lower for word in sql_words)
# --- End existing functions ---


# --- UPDATED Generate SQL Query Function ---
def generate_sql_query(user_query: str) -> str | None:
    """Generates the SQL query string using LLM, but does not execute it."""
    global db # Ensure we're using the global db object
    if not db:
        logger.error("SQLDatabase connection is not available for generating SQL.")
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model_name,
            temperature=0.0,
            # convert_system_message_to_human=True # Deprecated warning observed, removed. Let SDK handle if possible.
        )

        # --- Safely get the prompt template ---
        # Try PostgreSQL specific prompt first
        prompt_template = SQL_PROMPTS.get("postgres", None)

        if prompt_template is None:
             logger.warning("PostgreSQL prompt not found in SQL_PROMPTS. Using a generic fallback prompt.")
             # Define a generic prompt template string if specific one not found
             # This is similar to the Langchain default but defined locally
             GENERIC_SQL_TEMPLATE = """You are a {dialect} expert. Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data as per {dialect}.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which table contains which column.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}

Question: {question}
SQLQuery:""" # Modified to directly ask for SQLQuery
             prompt_template = PromptTemplate.from_template(GENERIC_SQL_TEMPLATE)
        # --- End prompt template selection ---


        # Extract table names and schema
        table_info = db.get_table_info(db.get_usable_table_names())
        dialect = db.dialect # Get dialect from db utility

        # Create the chain: Provide Schema+Question -> LLM -> Parse Output String
        sql_generation_chain = (
            RunnablePassthrough.assign(
                # Pass necessary inputs to the prompt template
                table_info=lambda _: table_info,
                question=lambda x: x["question"],
                dialect=lambda _: dialect,
                top_k=lambda _: 10 # Default top_k value, adjust if needed
            )
            | prompt_template         # Format the prompt
            | llm                     # Call the LLM
            | StrOutputParser()       # Get the string output
        )

        logger.info(f"Generating SQL for query: '{user_query[:50]}...'")
        # Invoke the chain to get the raw SQL string
        raw_sql_output = sql_generation_chain.invoke({"question": user_query})

        logger.info(f"Raw SQL output from LLM: {raw_sql_output}")
        # The output might still contain explanations or the expected 'Answer:' part.
        # We only want the SQL Query part. Refine extraction if needed.
        # For now, assume the primary output is the SQL, potentially with fences.
        return raw_sql_output

    except Exception as e:
        logger.error(f"Failed to generate SQL query: {e}", exc_info=True)
        return None
# --- End Generate SQL Query Function ---