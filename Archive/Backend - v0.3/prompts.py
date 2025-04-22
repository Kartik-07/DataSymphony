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



# You can add other prompt templates here (e.g., for HyDE, query decomposition)