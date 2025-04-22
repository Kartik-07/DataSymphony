# RAG_Project/Backend/generation.py

import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from prompts import RAG_PROMPT # Ensure RAG_PROMPT is defined appropriately

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Generates answers based on query and retrieved context using Google Gemini."""

    def __init__(self):
        """Initializes the LLM used for generation."""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.llm_model_name,
                temperature=0.1, # Low temperature for more factual RAG answers
                # Add other parameters like top_p, top_k if needed
            )
            logger.info(f"Initialized ChatGoogleGenerativeAI generator with model: {settings.llm_model_name}")
        except Exception as e:
             logger.error(f"Failed to initialize Google Gemini LLM ({settings.llm_model_name}) for generation: {e}", exc_info=True)
             # Depending on desired behavior, either raise the error or set self.llm to None
             # Setting to None allows checking availability later.
             self.llm = None
             # raise ValueError(f"Could not initialize main LLM: {e}") from e


    @staticmethod
    def _format_docs_with_metadata(docs: List[Document]) -> str:
        """
        Formats documents for the RAG prompt, including source metadata for citation.
        Uses the 'identifier' from metadata as the primary source name.
        """
        if not docs:
            return "No context documents found."

        formatted_docs = []
        for i, doc in enumerate(docs):
            # Use the 'identifier' (filename/table name) as the main citation source
            # Fallback to 'source' if identifier is missing for some reason
            source_name = doc.metadata.get('identifier', doc.metadata.get('source', f'Unknown Source {i+1}'))
            # Clean up source name slightly for display
            source_name = source_name.replace('_', ' ').strip()
            metadata_str = f"Source: {source_name}"

            # Use page_content which contains the text chunk or the metadata summary
            content = doc.page_content
            formatted_docs.append(f"--- Document {i+1} [{metadata_str}] ---\n{content}")

        return "\n\n".join(formatted_docs)

    # Note: get_rag_chain might not be directly used by the latest rag_pipeline logic,
    # but it's kept here as a standard way to represent the RAG chain structure.
    def get_rag_chain(self, retriever):
        """Creates a runnable RAG chain using LCEL (kept for reference)."""
        if not self.llm:
             raise RuntimeError("Generator LLM is not initialized.")

        format_context = RunnableLambda(self._format_docs_with_metadata)

        retrieve_and_format = RunnablePassthrough.assign(
            # The input to this chain segment is expected to be {"question": query}
            context=(lambda x: x["question"]) | retriever | format_context
        )

        rag_chain = (
            retrieve_and_format # Output includes {"question": query, "context": formatted_docs}
            | RAG_PROMPT        # Uses "question" and "context" keys
            | self.llm
            | StrOutputParser()
        )
        logger.info("RAG chain structure defined in AnswerGenerator.")
        return rag_chain

    def generate_answer(self, query: str, retrieved_docs: List[Document]) -> str:
        """
        Generates an answer using the LLM, query, and formatted context documents.
        This method is directly called by the RAG pipeline for the VECTOR_STORE path.
        """
        if not self.llm:
             logger.error("Cannot generate answer: Generator LLM is not initialized.")
             return "Sorry, the answer generation service is currently unavailable."

        if not retrieved_docs:
            logger.warning("No documents provided to generate_answer. Cannot generate contextual answer.")
            # Match the message used in rag_pipeline for consistency
            return "Could not find relevant information in the indexed documents to answer the query."

        # Format the context using the helper method
        context = self._format_docs_with_metadata(retrieved_docs)
        logger.debug(f"Formatted Context for LLM:\n{context[:500]}...") # Log beginning of context

        # Create the prompt input dictionary
        prompt_input = {"context": context, "question": query}
        logger.debug(f"Using RAG_PROMPT: {RAG_PROMPT.template[:100]}...") # Log template snippet

        # Define the generation part of the chain
        generation_chain = RAG_PROMPT | self.llm | StrOutputParser()

        try:
            # Invoke the generation chain
            answer = generation_chain.invoke(prompt_input)
            logger.info(f"Generated answer for query: '{query[:50]}...'")
            return answer
        except Exception as e:
            logger.error(f"Error during answer generation for query '{query[:50]}...': {e}", exc_info=True)
            return "Sorry, I encountered an error while generating the answer."