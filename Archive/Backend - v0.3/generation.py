# RAG_Project/MY_RAG/Backend/generation.py

import logging
from typing import List, Dict, Any, Tuple, Optional # Added Optional, Tuple

# --- Langchain Imports ---
from langchain_core.documents import Document
# Remove RunnableLambda, RunnablePassthrough if not used directly here
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
# Remove SystemMessage if only using PromptTemplate structure now
# from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate # Added

# --- Local Imports ---
# Import the necessary prompts and helpers from prompts.py
from prompts import (
    create_rag_prompt_template, # Function to get the dynamic template
    format_chat_history,        # Helper to format history string
    ANSWER_SUFFICIENCY_CHECK_PROMPT # Prompt for sufficiency check
    # FALLBACK_PROMPT might be needed if generation itself calls fallback? Unlikely based on pipeline structure.
)
from config import settings

# --- Import ChatMessage for Type Hinting ---
from models import ChatMessage


logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    Generates answers based on query and retrieved context using Google Gemini.
    Dynamically selects the appropriate prompt based on conversation history.
    Also uses a light LLM to check if the generated answer sufficiently addresses
    the query based *only* on the provided context.
    """

    def __init__(self):
        """Initializes the LLMs used for generation and sufficiency check."""
        self.llm = None # Main LLM for generation
        self.light_llm = None # Light LLM for sufficiency check

        # Initialize Main LLM (from settings.llm_model_name)
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.llm_model_name, # Main model
                temperature=0.1, # Low temperature for factual RAG
                convert_system_message_to_human=True # Often helpful for chat models with PromptTemplate
            )
            logger.info(f"Initialized main Generator LLM: {settings.llm_model_name}")
        except Exception as e:
             logger.error(f"Failed to init main Generator LLM ({settings.llm_model_name}): {e}", exc_info=True)

        # Initialize Light LLM (from settings.light_llm_model_name)
        try:
            if not settings.light_llm_model_name:
                 raise ValueError("LIGHT_LLM_MODEL_NAME is not set in config.")
            self.light_llm = ChatGoogleGenerativeAI(
                model=settings.light_llm_model_name, # Light model from config
                temperature=0.0, # Zero temperature for deterministic check
                convert_system_message_to_human=True # Add if needed for this model too
            )
            logger.info(f"Initialized light Sufficiency Check LLM: {settings.light_llm_model_name}")
        except Exception as e:
            logger.error(f"Failed to init light Sufficiency Check LLM ({settings.light_llm_model_name}): {e}", exc_info=True)
            self.light_llm = None


    @staticmethod
    def _format_docs_with_metadata(docs: List[Document]) -> str:
        """
        Formats documents for the RAG prompt, including source metadata for citation.
        Uses the 'identifier' from metadata as the primary source name.
        (Keep this method exactly as in your fetched code)
        """
        if not docs:
            return "No context documents found."

        formatted_docs = []
        for i, doc in enumerate(docs):
            source_name = doc.metadata.get('identifier', doc.metadata.get('source', f'Unknown Source {i+1}'))
            source_name = source_name.replace('_', ' ').strip()
            metadata_str = f"Source: {source_name}"
            content = doc.page_content
            formatted_docs.append(f"--- Document {i+1} [{metadata_str}] ---\n{content}")

        return "\n\n".join(formatted_docs)

    def _check_sufficiency_with_llm(self, query: str, context: str, answer_from_context: str) -> bool:
        """
        Uses the light LLM to check if the answer is sufficient based on context.
        (Keep this method exactly as in your fetched code)
        """
        if not self.light_llm:
            logger.warning("Light LLM for sufficiency check unavailable. Assuming answer is sufficient.")
            return True # Default to sufficient if checker is down

        if not answer_from_context or answer_from_context.startswith("Sorry, I encountered an error"):
             logger.warning("Initial answer was empty or an error message. Checking sufficiency skipped, marking as insufficient.")
             return False

        try:
            check_input = {
                "context": context,
                "question": query,
                "answer_from_context": answer_from_context
            }
            # Ensure ANSWER_SUFFICIENCY_CHECK_PROMPT is imported and valid
            sufficiency_chain = ANSWER_SUFFICIENCY_CHECK_PROMPT | self.light_llm | StrOutputParser()
            decision_str = sufficiency_chain.invoke(check_input).strip().upper()
            logger.info(f"Sufficiency Check LLM Decision: '{decision_str}'")
            # Check for the exact word "SUFFICIENT"
            return decision_str == "SUFFICIENT"
        except Exception as e:
            logger.error(f"Error during LLM sufficiency check: {e}", exc_info=True)
            # Default to sufficient on error to prevent fallback due to checker failure
            return True

    # --- MODIFIED generate_answer method ---
    def generate_answer(self,
                        query: str,
                        retrieved_docs: List[Document], # Kept name from original code
                        conversation_history: Optional[List[ChatMessage]] = None # Added history parameter
                       ) -> Tuple[str, bool]:
        """
        Generates an answer using the main LLM based on context and conversation history.
        Dynamically selects the prompt template based on history presence.
        Then uses the light LLM to check if that answer was sufficient given *only* the context.

        Args:
            query: The user's query string.
            retrieved_docs: List of documents retrieved for context.
            conversation_history: Optional list of previous chat messages.

        Returns:
            Tuple[str, bool]: The generated answer string and a boolean flag
                              indicating if the answer was deemed sufficient based
                              on the provided context (True=sufficient, False=insufficient).
        """
        initial_answer = ""
        is_sufficient = True # Default to sufficient

        # Check if main LLM is available
        if not self.llm:
             logger.error("Cannot generate answer: Main Generator LLM is not initialized.")
             return "Sorry, the answer generation service is currently unavailable.", False

        # Check if context was provided (same as original)
        if not retrieved_docs:
            logger.warning("No documents provided to generate_answer.")
            return "(No specific documents found to answer this query)", False

        # 1. Format context (same as original)
        try:
             context_str = self._format_docs_with_metadata(retrieved_docs)
        except Exception as fmt_err:
             logger.error(f"Error formatting documents: {fmt_err}", exc_info=True)
             return f"Error preparing context for generation: {fmt_err}", False

        # 2. Get Correct Prompt Template based on History
        try:
            prompt_template: PromptTemplate = create_rag_prompt_template(conversation_history)
            logger.debug(f"Using prompt template with input variables: {prompt_template.input_variables}")
        except Exception as prompt_err:
            logger.error(f"Error creating prompt template: {prompt_err}", exc_info=True)
            return f"Error setting up prompt: {prompt_err}", False

        # 3. Prepare Inputs for the LLM, including formatted history if needed
        llm_inputs = {
            "context": context_str,
            "question": query,
        }
        if "chat_history" in prompt_template.input_variables:
            # Format history only if the chosen template requires it
            formatted_history = format_chat_history(conversation_history)
            llm_inputs["chat_history"] = formatted_history
            logger.debug("Added formatted chat history to LLM inputs.")
        else:
            logger.debug("No chat history added to LLM inputs (template doesn't require it or history is empty).")


        # 4. Generate Initial Answer using Main LLM and the selected prompt
        try:
            generation_chain = prompt_template | self.llm | StrOutputParser()
            logger.debug(f"Invoking generation chain with inputs: { {k: (v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in llm_inputs.items()} }") # Log truncated inputs
            initial_answer = generation_chain.invoke(llm_inputs)
            logger.info(f"Generated initial answer for query: '{query[:50]}...'")

            # Handle empty answer from LLM (same as original)
            if not initial_answer:
                logger.warning("Main LLM returned an empty answer based on the context/history.")
                initial_answer = "(The language model did not provide an answer based on the provided documents and history.)" # Updated placeholder
                is_sufficient = False # Empty answer is insufficient

        except Exception as e:
            logger.error(f"Error during initial answer generation for query '{query[:50]}...': {e}", exc_info=True)
            return f"Sorry, I encountered an error while generating the initial answer: {e}", False

        # 5. Check Sufficiency using Light LLM (if available and initial generation didn't fail)
        # This check should *still* only use the retrieved context, not the history,
        # to evaluate if the answer *could* have been derived *solely* from the docs.
        if is_sufficient: # Only check if the answer wasn't already deemed insufficient (e.g., empty)
            logger.debug("Checking answer sufficiency based on retrieved context...")
            # Note: We pass context_str (formatted docs), not the formatted_history,
            # to the sufficiency check, as per its original purpose.
            is_sufficient = self._check_sufficiency_with_llm(query, context_str, initial_answer)

        # 6. Return the initial answer and the final sufficiency flag
        logger.debug(f"Returning generated answer. Sufficient: {is_sufficient}")
        return initial_answer, is_sufficient