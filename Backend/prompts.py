# RAG_Project/MY_RAG/Backend/prompts.py

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime


# Langchain imports for prompt templating
from langchain_core.prompts import PromptTemplate

# Import ChatMessage definition (adjust path if necessary, consider shared models.py)
from models import ChatMessage

logger = logging.getLogger(__name__)

# --- History Formatting Helper ---
def format_chat_history(history: Optional[List[ChatMessage]], max_turns: int = 5) -> str:
    """
    Formats the chat history into a string for the LLM prompt.
    Limits history to the last `max_turns` user/AI pairs.
    """
    if not history:
        return "No previous conversation history provided." # Return specific string

    max_messages = max_turns * 2
    start_index = max(0, len(history) - max_messages)
    recent_history = history[start_index:]

    formatted_history = []
    for msg in recent_history:
        sender = "User" if msg.sender == "user" else "AI"
        formatted_history.append(f"{sender}: {msg.text}")

    if not formatted_history:
         return "No recent conversation history to display."

    return "\n".join(formatted_history)

# Basic RAG Prompt - Used when context IS available and relevant
RAG_PROMPT_TEMPLATE_NO_HISTORY = """
CONTEXT:
{context}

QUERY:
{question}

INSTRUCTIONS:
Based *only* on the provided CONTEXT, answer the QUERY.
If the context does not contain the answer, state that the context is insufficient.

ANSWER:
"""


RAG_PROMPT_TEMPLATE_WITH_HISTORY = """
Chat History:
{chat_history}

Context:
{context}

QUERY:
{question}

INSTRUCTIONS:
Use prior conversation history as contextual reference to inform and enrich your response, but do not rely on it exclusively.

Using only the information provided in the Context and Chat History, answer the user’s query.

If the answer is unknown based on the available information, clearly state that you don't know—do not fabricate a response.

Be concise, accurate, and helpful.

Answer: """

# --- Function to Create the Appropriate PromptTemplate ---
def create_rag_prompt_template(
    history: Optional[List[ChatMessage]] = None
) -> PromptTemplate:
    """
    Creates the appropriate PromptTemplate based on whether history is present,
    following the structure of the original RAG_PROMPT_TEMPLATE.
    """
    if history and len(history) > 0:
        # History exists, use the template with history
        logger.debug("Creating RAG PromptTemplate with history.")
        return PromptTemplate(
            template=RAG_PROMPT_TEMPLATE_WITH_HISTORY,
            input_variables=["context", "question", "chat_history"]
        )
    else:
        # No history, use the template without history
        logger.debug("Creating RAG PromptTemplate without history.")
        return PromptTemplate(
            template=RAG_PROMPT_TEMPLATE_NO_HISTORY,
            input_variables=["context", "question"]
        )

# Fallback Prompt - Used when no relevant context is found or retrieval fails
FALLBACK_PROMPT_TEMPLATE = """
QUERY:
{question}

INSTRUCTIONS:
Please answer the QUERY based on your general knowledge. If the query is highly specific and likely requires external documents you don't have access to, state that you cannot answer accurately without specific context.

ANSWER:
"""

FALLBACK_PROMPT = PromptTemplate.from_template(FALLBACK_PROMPT_TEMPLATE)

# Fallback Prompt - Used when no relevant context is found BUT history is available
FALLBACK_PROMPT_WITH_HISTORY_TEMPLATE = """
Chat History:
{chat_history}

QUERY:
{question}

INSTRUCTIONS:
Use prior Chat history as contextual reference to inform and enrich responses, but don't rely on it exclusively. Maintain continuity while allowing for fresh ideas, shifts in tone, or new directions. Prioritize relevance, coherence, and natural conversation flow over strict adherence to past context.

If the user's query is highly specific and appears to require external documents or context not provided, clearly state that you cannot answer accurately without access to that specific information.

ANSWER:
"""

FALLBACK_PROMPT_WITH_HISTORY = PromptTemplate.from_template(FALLBACK_PROMPT_WITH_HISTORY_TEMPLATE)


# Prompt for checking if the RAG answer is sufficient
ANSWER_SUFFICIENCY_CHECK_PROMPT_TEMPLATE = """
Analyze the provided Query, Context, and the Answer generated *strictly* from that Context.
Determine if the Answer fully addresses the Query based *only* on the information present in the Context.

Context:
---
{context}
---

Query:
---
{question}
---

Answer Generated from Context:
---
{answer_from_context}
---

INSTRUCTIONS:
- If the "Answer Generated from Context" *fully* answers the "Query" using *only* information found in the "Context", respond with "SUFFICIENT".
- If the "Answer Generated from Context" correctly states limitations or cannot fully answer the "Query" because the necessary details are missing in the "Context", respond with "INSUFFICIENT".

Respond ONLY with the single word "SUFFICIENT" or "INSUFFICIENT".

Decision:
"""
ANSWER_SUFFICIENCY_CHECK_PROMPT = PromptTemplate.from_template(ANSWER_SUFFICIENCY_CHECK_PROMPT_TEMPLATE)



# --- *** NEW: Router Prompt Template *** ---
# This prompt will guide the LLM in rag_pipeline._route_query_llm
ROUTER_PROMPT_TEMPLATE = """
You are an expert data query routing assistant. Your task is to analyze the user's query and the available data sources to determine the single best execution path AND the primary data source if applicable.

Available Data Sources (Metadata and Summaries):
---
{available_sources}
---

User Query:
---
"{query_text}"
---

Instructions:
1. Examine the User Query carefully. Understand if it requires factual data retrieval, calculations, data analysis, plotting, comparisons, or general knowledge.
2. Review the Available Data Sources. Pay attention to ID, Type (structured/unstructured), Summary, Columns (for structured data).
3. Make a Decision based on the best way to fulfill the User Query:
    * Choose 'SQL:<Source ID>': If the query asks for specific data points, calculations, filtering, or aggregations that directly match a *single* structured source's Columns and Summary. Example: `SQL:sales_data_q1.xlsx`
    * Choose 'PYTHON_ANALYSIS:<Source ID>': If the query requires data manipulation, analysis (e.g., correlations, statistics beyond simple SQL aggregates), visualization/plotting on a specific structured source identified from 'Available Data Sources'. Example: `PYTHON_ANALYSIS:sales_data.xlsx`
    * Choose 'PYTHON_ANALYSIS': If the query requires Python execution but doesn't clearly operate on one specific source from the list (e.g., "plot sin(x)"), or if it needs data from *multiple* sources (which Python will handle). Example: `PYTHON_ANALYSIS`
    * Choose 'VECTOR_STORE': If the query is general, asks for definitions, concepts, summaries from unstructured text, or if the answer is likely contained within the text content retrieved from vector search, and does *not* require complex Python analysis or specific SQL lookups. Example: `VECTOR_STORE`
    * Choose 'NONE': If no available data source seems relevant to the query and it cannot be answered by general knowledge or analysis. Example: `NONE`

4. Output Format: Respond ONLY with the chosen decision (e.g., `SQL:sales_data_q1.xlsx`, `PYTHON_ANALYSIS:sales_data.xlsx`, `PYTHON_ANALYSIS`, `VECTOR_STORE`, `NONE`).
Decision:
"""
ROUTER_PROMPT = PromptTemplate(
    template=ROUTER_PROMPT_TEMPLATE,
    input_variables=["available_sources", "query_text"]
)


# --- *** NEW: Python Code Generation Prompt Template *** ---
# This prompt will be used by the LLM in rag_pipeline.py before calling the executor
PYTHON_GENERATION_PROMPT_TEMPLATE = """
You are an expert Python data scientist. Your task is to write Python code to answer the user's query based on the provided information about a pandas DataFrame (`df`).

User Query:
---
{question}
---

Available Data Information (Metadata for DataFrame `df`):
---
{data_info}
---
(This section contains metadata about the DataFrame 'df' that has been loaded for you, including description, row count, and column details like Name, Type, and Description.)

Chat History (for context):
---
{chat_history}
---

Instructions for Code Generation:

**1. Understand the Goal:**
   - Write Python code using the pre-loaded DataFrame `df` to directly address the User Query.
   - Focus *only* on answering the query based on the provided data and metadata.

**2. Available Tools:**
   - **Libraries:** Assume `pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, `scipy`, and `tabulate` are imported and available. Do NOT add imports for these. `matplotlib.use('Agg')` is already set.
   - **DataFrame:** A pandas DataFrame named `df` is pre-loaded if 'Available Data Information' describes it. If not, proceed based on general knowledge using only the allowed libraries.

**3. Data Handling (CRITICAL):**
   - **Use Metadata:** Refer *strictly* to the 'Available Data Information' provided above for DataFrame structure.
   - **Column Names:** Use the **EXACT** column names listed in the metadata (e.g., `df['Weekly_Sales']`, `df['Date']`). Do NOT assume names based on the query (e.g., do not use 'Total Revenue' if the column is 'Weekly_Sales').
   - **Data Types:** Pay attention to the column 'Type' in the metadata. Apply operations suitable for that type (e.g., sum numerical columns, parse date columns).
   - **Date Parsing:**
      - When converting date columns identified in metadata, use `pd.to_datetime(...)`.
      - Specify the format string explicitly if known from metadata or examples (e.g., `format='%d-%m-%Y'`, `format='%Y-%m-%d'`).
      - If the format is unclear, try `format='mixed', dayfirst=True` or `infer_datetime_format=True`.
      - **ALWAYS include `errors='coerce'`** (e.g., `pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')`) to prevent crashes on unparseable dates. Assign the result back, e.g., `df['Date'] = pd.to_datetime(...)`.

**4. Code Output:**
   - **Print Results:** Use `print()` to output any final answers, calculations, summaries, or relevant information requested by the query.
   - **Table Formatting:** For DataFrame or list-of-dict results, consider using `tabulate` for clear output: `print(tabulate(my_result_df, headers='keys', tablefmt='psql'))`.
   - **Plotting:** If plotting is requested:
      - Generate the plot using `matplotlib.pyplot`.
      - Ensure plots have titles and axis labels (`plt.title(...)`, `plt.xlabel(...)`, `plt.ylabel(...)`).
      - **Save the plot EXACTLY as `plot.png`**: `plt.savefig('plot.png')`.
      - **Do NOT use `plt.show()`**.

**5. Constraints & Safety:**
   - Write only the Python code needed for the task. Do not add surrounding text or explanations.
   - Do NOT import `os`, `subprocess`, `sys`, or any other system-interacting libraries.
   - Do NOT read from or write to any files except saving plots to `plot.png`.
   - Stick to operations using the provided `df` and allowed libraries.

Write only the Python code required to answer the query.

Python Code:
```python
# Your Python code starts here
{initialization_code} # Code to initialize 'df' if data was loaded

# --- Debug: Print DataFrame Info ---
if 'df' in locals() and isinstance(df, pd.DataFrame):
    print("--- DataFrame Info ---")
    print(f"Columns: {{df.columns.tolist()}}")
    # print(df.info()) # df.info() prints directly, might be too verbose
    print("----------------------")
# --- End Debug ---

# --- User Query Analysis Code ---
{user_code_instructions} # Placeholder, the main code will go after initialization
"""
PYTHON_GENERATION_PROMPT = PromptTemplate(
template=PYTHON_GENERATION_PROMPT_TEMPLATE,
input_variables=["question", "data_info", "chat_history", "initialization_code", "user_code_instructions"] # Added placeholders
)