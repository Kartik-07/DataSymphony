import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI <-- Remove
from langchain_google_genai import ChatGoogleGenerativeAI # <-- Add

from config import settings
from prompts import RAG_PROMPT

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Generates answers based on query and retrieved context using Google Gemini."""

    def __init__(self):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.llm_model_name,
                temperature=0.1, # Adjust temperature as needed
                # google_api_key=settings.google_api_key, # Implicitly read from env
                # convert_system_message_to_human=True # Often helpful for Gemini's chat handling
            )
            logger.info(f"Initialized ChatGoogleGenerativeAI with model: {settings.llm_model_name}")
        except Exception as e:
             logger.error(f"Failed to initialize Google Gemini LLM: {e}", exc_info=True)
             raise

    @staticmethod
    def _format_docs_with_metadata(docs: List[Document]) -> str:
        """Formats documents for the prompt, including metadata for citation."""
        formatted_docs = []
        for i, doc in enumerate(docs):
            metadata_str = f"Source: {doc.metadata.get('source', 'N/A')}"
            # Add other relevant metadata if needed, e.g., chunk_id
            # metadata_str += f" | Chunk: {doc.metadata.get('chunk_id', 'N/A')}"
            formatted_docs.append(f"--- Document {i+1} [{metadata_str}] ---\n{doc.page_content}")
        return "\n\n".join(formatted_docs) if formatted_docs else "No context documents found."

    def get_rag_chain(self, retriever):
        """Creates the RAG chain using LCEL."""

        # Runnable to format context
        format_context = RunnableLambda(self._format_docs_with_metadata)

        # Context retrieval step
        # Takes the input dict {"question": query}, retrieves docs, formats them
        retrieve_and_format = RunnablePassthrough.assign(
            context=(lambda x: x["question"]) | retriever | format_context
        )
        # RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_context) # alternative

        # Full RAG chain
        rag_chain = (
            retrieve_and_format
            | RAG_PROMPT # Takes dict with "question" and "context"
            | self.llm
            | StrOutputParser()
        )
        logger.info("RAG chain created.")
        return rag_chain

    def generate_answer(self, query: str, retrieved_docs: List[Document]) -> str:
        """Generates an answer using the LLM, query, and context."""
        if not retrieved_docs:
            logger.warning("No documents retrieved. Cannot generate contextual answer.")
            # Optionally call LLM without context, or return specific message
            # return self.llm.invoke(f"Answer the following question: {query}").content
            return "I couldn't find relevant information to answer your question."

        context = self._format_docs_with_metadata(retrieved_docs)
        logger.debug(f"Formatted Context for LLM:\n{context[:500]}...") # Log beginning of context

        # Create the final prompt using the template
        prompt_input = {"context": context, "question": query}
        final_prompt = RAG_PROMPT.invoke(prompt_input) # Render the prompt
        logger.debug(f"Final Prompt for LLM:\n{final_prompt.to_string()[:500]}...")

        # Define the generation chain (without retriever, as docs are pre-fetched)
        generation_chain = RAG_PROMPT | self.llm | StrOutputParser()

        try:
            answer = generation_chain.invoke(prompt_input)
            logger.info(f"Generated answer for query: '{query[:50]}...'")
            return answer
        except Exception as e:
            logger.error(f"Error during answer generation for query '{query[:50]}...': {e}", exc_info=True)
            return "Sorry, I encountered an error while generating the answer."