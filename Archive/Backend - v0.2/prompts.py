from langchain_core.prompts import PromptTemplate

# Basic RAG Prompt
RAG_PROMPT_TEMPLATE = """
CONTEXT:
{context}

QUERY:
{question}

INSTRUCTIONS:
Based *only* on the provided CONTEXT, answer the QUERY.
If the context does not contain the answer, state that the context is insufficient.

ANSWER:
"""

RAG_PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# You can add other prompt templates here (e.g., for HyDE, query decomposition)