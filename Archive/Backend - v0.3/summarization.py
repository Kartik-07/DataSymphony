# RAG_Project/Backend/summarization.py

import json
import logging
import uuid
import os
import re
import warnings
from typing import Union, Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy

# Attempt to import necessary LangChain and utility components
try:
    from config import settings
    from langchain_core.documents import Document
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from sql_processing import get_sql_database_utility, SQLDatabase
    from data_processing import DataLoader
except ImportError as e:
    print(f"Warning: Summarization - Required LangChain/Project components not found: {e}. Using dummy classes.")
    # Define dummy classes if imports fail (for basic structure)
    class DummySettings:
        light_llm_model_name = "gemini-1.5-flash"
        google_api_key = os.getenv("GOOGLE_API_KEY")
    settings = DummySettings()
    class Document: pass
    class ChatGoogleGenerativeAI: pass
    class HumanMessage: pass
    class SystemMessage: pass
    class StrOutputParser: pass
    class SQLDatabase: pass
    def get_sql_database_utility() -> Optional[SQLDatabase]: return None
    class DataLoader: # Dummy DataLoader
        @staticmethod
        def load_pdf(path: str) -> List: return []
        @staticmethod
        def load_text(path: str) -> List: return []
        @staticmethod
        def load_docx(path: str) -> List: return []

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): logging.basicConfig(level=logging.INFO)

# --- Helper Functions ---
# (Keep clean_code_snippet and read_dataframe as before)
def clean_code_snippet(snippet: str) -> str:
    if not isinstance(snippet, str): return ""
    snippet = re.sub(r"```(?:[a-zA-Z0-9]*)?\s*(.*)\s*```", r"\1", snippet, flags=re.DOTALL)
    return snippet.strip()

def read_dataframe(path: str, encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
    logger.debug(f"Reading DataFrame from: {path}")
    try:
        if path.lower().endswith(('.xlsx', '.xls')): return pd.read_excel(path)
        elif path.lower().endswith('.csv'): return pd.read_csv(path, encoding=encoding)
        else: logger.warning(f"Unsupported type for DataFrame reading: {path}"); return None
    except FileNotFoundError: logger.error(f"File not found: {path}"); return None
    except ImportError as ie: logger.error(f"Missing dependency for {path}: {ie}"); raise ie
    except Exception as e: logger.error(f"Read failed {path}: {e}"); return None

# --- LLM Prompts ---
# (Keep STRUCTURED_ENRICHMENT_SYSTEM_PROMPT and UNSTRUCTURED_SUMMARY_SYSTEM_PROMPT as before)
STRUCTURED_ENRICHMENT_SYSTEM_PROMPT = """
You are an experienced data analyst that can annotate datasets based on their structure and sample values. Your instructions are as follows:
i) Review the dataset name (which corresponds to a database table name or filename) and column properties provided.
ii) Generate a concise, accurate `dataset_description` (1-2 sentences) based on the dataset name and columns.
iii) For each field (`column`) in the `fields` list:
    - Generate an accurate `semantic_type` (a single common noun or concept, e.g., company, city, count, product_id, location, gender, latitude, url, category, timestamp, identifier, measurement, description, etc.) based on the column name, data type (`dtype`), and sample values (`samples`).
    - Generate a brief `description` explaining the field's likely content or purpose within the table/file.
iv) Respond ONLY with the updated JSON dictionary, ensuring all original fields are present. Do not include any preamble, explanation, or markdown formatting.
"""
UNSTRUCTURED_SUMMARY_SYSTEM_PROMPT = """
You are an expert text analyst. Your task is to analyze the provided text content and extract key information.
Instructions:
1. Read the text content carefully.
2. Generate a concise, natural language `summary` (3-5 sentences) of the main topics and purpose of the text.
3. Identify the main `domain` or `theme` of the document (e.g., 'Finance', 'Healthcare', 'Technology', 'Legal', 'News Report', 'Scientific Article').
4. Extract relevant `keywords` (provide a list of 5-10 important terms or concepts).
5. If discernible from the text, identify the `author` and relevant `timestamp` (like creation or publication date). If not found, use "N/A".
6. If the document has a clear hierarchical structure (e.g., chapters, sections, headings), list the main `headings`. If not applicable, use an empty list [].
7. Respond ONLY with a JSON object containing the following keys: "summary", "domain", "keywords", "author", "timestamp", "headings". Do not include any preamble, explanation, or markdown formatting.
"""

# === Summarizer Class ===
class DataSummarizer():
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.llm_model = settings.light_llm_model_name
        effective_api_key = api_key or settings.google_api_key
        self.text_gen: Optional[ChatGoogleGenerativeAI] = None
        if not effective_api_key: logger.warning(f"Google API Key missing. LLM calls ({self.llm_model}) may fail.")
        else:
            try: self.text_gen = ChatGoogleGenerativeAI(model=self.llm_model, temperature=0.15)
            except Exception as e: logger.error(f"Failed to init Summarizer LLM: {e}"); self.text_gen = None
        self.db_utility: Optional[SQLDatabase] = get_sql_database_utility()
        logger.info(f"Summarizer Initialized (LLM: {'OK' if self.text_gen else 'Failed'}, DB Util: {'OK' if self.db_utility else 'Failed'})")


    # --- Helper methods for structured data --- (_check_type, _get_common_properties, _get_column_properties)
    # (Keep these helpers as they were in the previous version)
    def _check_type(self, dtype: Any, value: Any) -> Any: # ... same ...
        if pd.isna(value): return None
        try:
            if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
            elif isinstance(value, (np.integer)): return int(value)
            elif isinstance(value, (np.floating)): return float(value) if np.isfinite(value) else None
            elif isinstance(value, (np.bool_)): return bool(value)
            elif "float" in str(dtype): return float(value) if pd.notna(value) and np.isfinite(value) else None
            elif "int" in str(dtype): return int(value) if pd.notna(value) else None
            elif "bool" in str(dtype): return bool(value) if pd.notna(value) else None
            elif isinstance(value, (list, dict)): return value
            else: return str(value)
        except (ValueError, TypeError): return str(value)

    def _get_common_properties(self, series: pd.Series, n_samples: int) -> dict: # ... same ...
         properties = {}
         try:
             non_null_series = series.dropna(); properties["num_unique_values"] = int(non_null_series.nunique())
             non_null_values = non_null_series.unique(); actual_n_samples = min(n_samples, len(non_null_values))
             if actual_n_samples > 0:
                 sampled_indices = np.random.choice(len(non_null_values), actual_n_samples, replace=False)
                 samples = [non_null_values[i] for i in sampled_indices]
                 properties["samples"] = [self._check_type(series.dtype, s) for s in samples]
             else: properties["samples"] = []
         except Exception as e: logger.warning(f"Error common props '{series.name}': {e}"); properties["num_unique_values"], properties["samples"] = 0, []
         properties["semantic_type"], properties["description"] = "", ""
         return properties

    def _get_column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> list[dict]: # ... same ...
        properties_list = []
        if df.empty: return []
        for column in df.columns:
            col_name_str = str(column); logger.debug(f"Analyzing column: '{col_name_str}'")
            series = df[col_name_str]; dtype_orig = series.dtype
            properties = self._get_common_properties(series, n_samples)
            numeric_stats = {"std": None, "min": None, "max": None, "mean": None, "median": None, "p25": None, "p75": None}
            dt_stats = {"min": None, "max": None}
            try: # Stat calculation logic remains the same
                if pd.api.types.is_numeric_dtype(dtype_orig) and not pd.api.types.is_bool_dtype(dtype_orig):
                    properties["dtype"] = "number"; numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                    if not numeric_series.empty:
                         desc = numeric_series.describe(); numeric_stats.update({"std": self._check_type(dtype_orig, desc.get('std')), "min": self._check_type(dtype_orig, desc.get('min')), "max": self._check_type(dtype_orig, desc.get('max')), "mean": self._check_type(dtype_orig, desc.get('mean')), "median": self._check_type(dtype_orig, numeric_series.median()), "p25": self._check_type(dtype_orig, desc.get('25%')), "p75": self._check_type(dtype_orig, desc.get('75%'))})
                    properties.update(numeric_stats)
                elif pd.api.types.is_bool_dtype(dtype_orig): properties["dtype"] = "boolean"
                elif pd.api.types.is_datetime64_any_dtype(dtype_orig) or pd.api.types.is_timedelta64_dtype(dtype_orig):
                    properties["dtype"] = "datetime"; datetime_series = pd.to_datetime(series, errors='coerce').dropna()
                    if not datetime_series.empty: dt_stats.update({"min": self._check_type(dtype_orig, datetime_series.min()), "max": self._check_type(dtype_orig, datetime_series.max())})
                    properties.update(dt_stats)
                elif pd.api.types.is_categorical_dtype(series): properties["dtype"] = "category"; properties["samples"] = [str(s) for s in properties.get("samples", [])]
                elif dtype_orig == object or pd.api.types.is_string_dtype(dtype_orig):
                    try: # Datetime detection logic remains
                        with warnings.catch_warnings(): warnings.simplefilter("ignore"); non_null_sample = series.dropna().sample(min(50, series.count()), random_state=42) if series.count() > 0 else pd.Series(dtype=object)
                        if not non_null_sample.empty and pd.to_datetime(non_null_sample, errors='coerce').notna().all():
                            properties["dtype"] = "datetime"; datetime_series = pd.to_datetime(series, errors='coerce').dropna()
                            if not datetime_series.empty: dt_stats.update({"min": self._check_type(dtype_orig, datetime_series.min()), "max": self._check_type(dtype_orig, datetime_series.max())})
                            properties.update(dt_stats)
                        else: raise ValueError("Not datetime")
                    except: # Fallback
                        non_null_count = series.count(); unique_ratio = series.nunique() / non_null_count if non_null_count > 0 else 0
                        properties["dtype"] = "category" if unique_ratio < 0.6 else "string"
                    properties["samples"] = [str(s) for s in properties.get("samples", [])]
                else: properties["dtype"] = str(dtype_orig); properties["samples"] = [str(s) for s in properties.get("samples", [])]
            except Exception as stat_err: logger.warning(f"Stats/type error '{col_name_str}': {stat_err}"); properties.setdefault("dtype", str(dtype_orig))
            try: properties["missing_values_count"] = int(series.isnull().sum()); properties["missing_values_proportion"] = float(series.isnull().mean()) if len(series) > 0 else 0.0
            except Exception as common_stat_err: logger.warning(f"Common stats error '{col_name_str}': {common_stat_err}"); properties["missing_values_count"], properties["missing_values_proportion"] = -1, -1.0
            properties_list.append({"column": col_name_str, **properties}); logger.debug(f"Col '{col_name_str}' props: {properties}")
        return properties_list

    # --- LLM Interaction Methods (Synchronous) ---
    def _enrich_structured_summary(self, base_summary: dict) -> dict:
        # (Keep sync version using invoke as defined previously)
        if not self.text_gen: logger.warning("LLM unavailable. Skip enrichment."); base_summary.setdefault("metadata", {})["enrichment_status"] = "skipped_no_llm"; return base_summary
        identifier = base_summary.get('identifier', 'unknown_structured'); logger.info(f"Enriching structured: '{identifier}'")
        prompt_input_summary = {"name": base_summary.get("name"), "identifier": identifier, "dataset_description": base_summary.get("dataset_description", ""), "fields": [{"column": f.get("column"), "dtype": f.get("dtype"), "samples": f.get("samples", [])[:3]} for f in base_summary.get("fields", []) if f.get("column")] }
        prompt_content = json.dumps(prompt_input_summary, indent=2, default=str)
        lc_messages = [SystemMessage(content=STRUCTURED_ENRICHMENT_SYSTEM_PROMPT), HumanMessage(content=f"Annotate:\n{prompt_content}")]
        response_text = ""
        try:
            response = self.text_gen.invoke(lc_messages); response_text = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"LLM enrich response:\n{response_text}"); json_string = clean_code_snippet(response_text); enriched_data = json.loads(json_string)
            if not isinstance(enriched_data, dict): raise ValueError("LLM response not dict.")
            final_summary = base_summary.copy()
            if enriched_data.get("dataset_description"): final_summary["dataset_description"] = enriched_data["dataset_description"]
            enriched_fields_map = {str(f.get("column")): f for f in enriched_data.get("fields", []) if f.get("column")}
            updated_fields = []
            for field in final_summary.get("fields", []):
                enriched_props = enriched_fields_map.get(str(field["column"]))
                if enriched_props:
                     if enriched_props.get("semantic_type"): field["semantic_type"] = enriched_props["semantic_type"]
                     if enriched_props.get("description"): field["description"] = enriched_props["description"]
                updated_fields.append(field)
            final_summary["fields"] = updated_fields; final_summary.setdefault("metadata", {})["enrichment_status"] = "success"
            logger.info(f"Enriched structured: '{identifier}'.")
            return final_summary
        except Exception as e: error_msg = f"LLM enrich fail: {e}. Raw: '{response_text}'"; logger.error(error_msg)
        base_summary.setdefault("metadata", {}); base_summary["metadata"]["enrichment_status"] = "failed"; base_summary["metadata"]["enrichment_error"] = error_msg
        return base_summary

    def _generate_llm_summary_from_content(self, content: str, identifier: str, doc_type: str) -> Dict[str, Any]:
        # (Keep sync version using invoke as defined previously)
        if not self.text_gen: logger.warning("LLM unavailable. Cannot gen summary."); return {"identifier": identifier, "document_type": doc_type, "summary": "LLM unavailable.", "domain": "N/A", "keywords": [], "author": "N/A", "timestamp": "N/A", "headings": [], "metadata": {"llm_status": "skipped_no_llm"}}
        logger.info(f"Generating LLM summary: {identifier} (len: {len(content)})")
        MAX_CONTENT_LENGTH = 750000
        content_to_send = (content[:MAX_CONTENT_LENGTH] + "\n[Truncated]") if len(content) > MAX_CONTENT_LENGTH else content
        if len(content) > MAX_CONTENT_LENGTH: logger.warning(f"Content truncated: {identifier}")
        lc_messages = [ SystemMessage(content=UNSTRUCTURED_SUMMARY_SYSTEM_PROMPT), HumanMessage(content=f"Analyze:\n\n{content_to_send}") ]
        response_text = ""
        try:
            response = self.text_gen.invoke(lc_messages); response_text = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"LLM unstructured summary response:\n{response_text}"); json_string = clean_code_snippet(response_text); llm_extracted_data = json.loads(json_string)
            required_keys = ["summary", "domain", "keywords", "author", "timestamp", "headings"]
            for key in required_keys: llm_extracted_data.setdefault(key, "N/A" if key in ["author", "timestamp", "domain", "summary"] else [])
            result = {"identifier": identifier, "document_type": doc_type, **llm_extracted_data, "metadata": {"llm_status": "success"}}
            logger.info(f"Generated unstructured summary: {identifier}.")
            return result
        except Exception as e: error_msg = f"LLM unstructured summary fail: {e}. Raw: '{response_text}'"; logger.error(error_msg)
        return {"identifier": identifier, "document_type": doc_type, "error": error_msg, "metadata": {"llm_status": "failed"}}

    # --- Formatting Output ---
    # (Keep _format_output_json as defined previously)
    def _format_output_json(self, summary_data: Dict[str, Any], data_type: str, source_type: str, target_db: Optional[str] = None, target_table: Optional[str] = None) -> Dict[str, Any]:
         identifier = summary_data.get("identifier", "unknown_source")
         safe_identifier = re.sub(r'[^\w.\-]+', '_', str(identifier)); unique_suffix = str(uuid.uuid4())[:8]
         output_id = f"{data_type}-{safe_identifier}-{unique_suffix}"
         output = { "id": output_id, "document": "", "metadata": {} }
         current_time_utc = datetime.now(timezone.utc).isoformat()
         output["metadata"].update({"identifier": identifier, "data_type": data_type, "source_type": source_type, "collection_time": current_time_utc})
         if data_type == "error": output["document"] = f"Error: {identifier}."; output["metadata"]["error"] = summary_data.get("error", "Unknown")
         elif data_type == "structured":
             output["document"] = summary_data.get("dataset_description", f"Structured summary: {identifier}.")
             if not output["document"]: output["document"] = f"Data: {identifier}. Rows: {summary_data.get('row_count')}, Cols: {summary_data.get('column_count')}."
             output["metadata"].update({ "row_count": summary_data.get("row_count"), "column_count": summary_data.get("column_count"), "columns": summary_data.get("fields", []), "enrichment_status": summary_data.get("metadata", {}).get("enrichment_status", "not_applicable"), "enrichment_error": summary_data.get("metadata", {}).get("enrichment_error"), "target_database": target_db, "target_table_name": target_table })
             if output["metadata"]["enrichment_status"] in ["success", "not_applicable"]: output["metadata"].pop("enrichment_error", None)
             if not target_db: output["metadata"].pop("target_database", None)
             if not target_table: output["metadata"].pop("target_table_name", None)
         elif data_type == "unstructured":
             output["document"] = summary_data.get("summary", f"Unstructured summary: {identifier}.")
             if not output["document"] or output["document"] == "LLM unavailable.": output["document"] = f"Content: {identifier}. Type: {summary_data.get('document_type')}."
             output["metadata"].update({ "document_type": summary_data.get("document_type"), "domain_themes": summary_data.get("domain"), "keywords": summary_data.get("keywords", []), "author": summary_data.get("author"), "timestamp": summary_data.get("timestamp"), "headings": summary_data.get("headings", []), "llm_status": summary_data.get("metadata", {}).get("llm_status") })
             if "error" in summary_data and output["metadata"]["llm_status"] and output["metadata"]["llm_status"].startswith("failed"): output["metadata"]["llm_error"] = summary_data.get("error")
         else: output["document"] = f"Summary: {identifier}."; output["metadata"]["error"] = "Unknown type"; output["metadata"]["data_type"] = "unknown"
         def make_serializable(obj): # Simplified helper
             if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
             elif isinstance(obj, Path): return str(obj)
             elif pd.isna(obj): return None
             try: json.dumps(obj); return obj
             except TypeError: return str(obj)
         try: output["metadata"] = json.loads(json.dumps(output["metadata"], default=make_serializable))
         except Exception as json_err: logger.error(f"JSON serialization error {identifier}: {json_err}"); output["metadata"]["json_error"] = str(json_err)
         output["metadata"] = {k: v for k, v in output["metadata"].items() if v is not None}
         return output

    # --- Main Summarize Method (Synchronous) ---
    def summarize( # Sync def
            self, data_input: Union[pd.DataFrame, str, None] = None, table_name: Optional[str] = None,
            file_name_override: Optional[str] = None, n_samples: int = 3, summary_method: str = "auto",
            encoding: str = 'utf-8', table_row_limit: Optional[int] = 1000, preloaded_content: Optional[str] = None,
            target_db: Optional[str] = None, target_table: Optional[str] = None) -> dict:
        """Summarize data from various sources synchronously."""
        data_type = "unknown"; summary_result = {}; identifier = None; df_to_process = None
        effective_method = summary_method; source_type = "unknown"

        # --- Determine Input Source ---
        if table_name:
            identifier = table_name; source_type = "database_table"; data_type = "structured"
            effective_method = 'llm' if summary_method in ['auto', 'llm'] else 'default'
            logger.info(f"Processing DB table: '{identifier}' method '{effective_method}'.")
            if not self.db_utility: summary_result = {"error": "RAG_DB utility missing."}; data_type = "error"
            else:
                try:
                    # --- FIX: Use _engine ---
                    engine = self.db_utility._engine # Use the correct attribute name
                    inspector = sqlalchemy.inspect(engine)
                    schema = None; actual_table_name = table_name
                    if '.' in table_name: schema, actual_table_name = table_name.split('.', 1)
                    if not inspector.has_table(actual_table_name, schema=schema): raise ValueError(f"Table '{table_name}' not found.")
                    sql_query = f'SELECT * FROM {self.db_utility.dialect_specific_quote(table_name)}' # dialect_specific_quote is hypothetical, adjust if needed
                    if table_row_limit: sql_query += f" LIMIT {table_row_limit}"
                    logger.info(f"Executing query: {sql_query}")
                    # Use the engine obtained above for the connection
                    with engine.connect() as connection: # Use the engine's connect method
                         df_to_process = pd.read_sql_query(sql=sqlalchemy.text(sql_query), con=connection) # Sync read

                    logger.info(f"Fetched {len(df_to_process)} rows: '{table_name}'.")
                    if df_to_process.empty:
                         columns_info = inspector.get_columns(actual_table_name, schema=schema)
                         base_summary = {"identifier": identifier, "name": identifier, "dataset_description": "Empty DB table.", "fields": [{"column": str(c['name']), "dtype": str(c['type'])} for c in columns_info], "row_count": 0, "column_count": len(columns_info)}
                         summary_result = self._enrich_structured_summary(base_summary) if effective_method == 'llm' and self.text_gen else base_summary
                    else: df_to_process.columns = [str(col) for col in df_to_process.columns]
                except AttributeError as ae: # Specifically catch if _engine isn't found either
                     logger.error(f"Attribute error accessing DB engine/inspector for table '{table_name}': {ae}", exc_info=True)
                     summary_result = {"error": f"DB attribute error: {ae}"}; data_type = "error"
                except Exception as db_err: logger.error(f"Table error '{table_name}': {db_err}"); summary_result = {"error": f"DB error: {db_err}"}; data_type = "error"
        # --- Rest of the input source handling logic remains the same ---
        elif preloaded_content is not None:
             source_type = "file"; identifier = file_name_override or f"preloaded_{str(uuid.uuid4())[:4]}"
             logger.info(f"Processing preloaded: '{identifier}'.")
             if summary_method == 'auto': effective_method = 'unstructured'
             if effective_method == 'unstructured': data_type = "unstructured"
             elif effective_method in ['llm', 'default']:
                  data_type = "structured"; logger.warning(f"Attempting structured summary on preloaded: {identifier}.")
                  try:
                      from io import StringIO
                      if '\n' in preloaded_content and ',' in preloaded_content.split('\n', 1)[0]:
                          df_to_process = pd.read_csv(StringIO(preloaded_content))
                          if df_to_process.empty: summary_result = {"identifier": identifier, "name": identifier, "dataset_description": "Empty preloaded (CSV).", "fields": [], "row_count": 0, "column_count": 0}
                          else: df_to_process.columns = [str(col) for col in df_to_process.columns]
                      else: raise ValueError("Preloaded not CSV-like")
                  except Exception as parse_err: logger.error(f"Parse fail: {parse_err}"); summary_result = {"error": f"Parse fail: {parse_err}"}; data_type = "error"
             else: summary_result = {"error": f"Unsupported method '{effective_method}'"}; data_type = "error"
        elif isinstance(data_input, str): # File path
            file_path = data_input; source_type = "file"; identifier = file_name_override or os.path.basename(file_path)
            logger.info(f"Processing file path: '{identifier}'.")
            if not os.path.exists(file_path): summary_result = {"error": "File not found"}; data_type = "error"
            else:
                file_ext = os.path.splitext(identifier)[1].lower()
                if summary_method == 'auto': effective_method = 'llm' if file_ext in ['.csv', '.xlsx', '.xls', '.parquet'] else 'unstructured'
                if effective_method in ['llm', 'default']:
                     data_type = "structured"
                     try:
                         df = read_dataframe(file_path, encoding=encoding)
                         if df is None: raise ValueError("read_dataframe failed.")
                         elif df.empty: summary_result = {"identifier": identifier, "name": identifier, "dataset_description": "Empty file.", "fields": [], "row_count": 0, "column_count": 0}
                         else: df_to_process = df; df_to_process.columns = [str(col) for col in df_to_process.columns]
                     except Exception as load_err: logger.error(f"Load error: {load_err}"); summary_result = {"error": f"Load fail: {load_err}"}; data_type = "error"
                elif effective_method == 'unstructured': data_type = "unstructured"
                else: summary_result = {"error": f"Invalid method '{effective_method}'"}; data_type = "error"
        elif isinstance(data_input, pd.DataFrame): # DataFrame
            identifier = file_name_override or "dataframe_input"; source_type = "dataframe"; data_type = "structured"
            if summary_method == 'unstructured': summary_result = {"error": "Unstructured needs file."}; data_type = "error"
            elif data_input.empty: summary_result = {"identifier": identifier, "name": identifier, "dataset_description": "Empty DataFrame.", "fields": [], "row_count": 0, "column_count": 0}
            else: df_to_process = data_input.copy(); df_to_process.columns = [str(col) for col in df_to_process.columns]; effective_method = 'llm' if summary_method in ['auto', 'llm'] else 'default'
        else: summary_result = {"error": "Invalid input type."}; data_type = "error"; identifier = "invalid_input"

        # --- Perform Summarization (Sync) ---
        if data_type != "error" and summary_result.get("row_count", -1) != 0:
             if data_type == "structured" and df_to_process is not None:
                 logger.debug(f"Running structured summarization: {identifier}")
                 try:
                     column_properties = self._get_column_properties(df_to_process, n_samples)
                     base_summary = {"identifier": identifier, "name": identifier, "dataset_description": "", "fields": column_properties, "row_count": len(df_to_process), "column_count": len(df_to_process.columns)}
                     if effective_method == "llm" and self.text_gen: summary_result = self._enrich_structured_summary(base_summary)
                     else:
                          summary_result = base_summary; summary_result.setdefault("metadata", {})["enrichment_status"] = "not_applicable"
                          if not summary_result.get("dataset_description"): summary_result["dataset_description"] = f"Basic stats: {identifier}."
                 except Exception as e: logger.error(f"Structured summary failed: {e}"); summary_result = {"identifier": identifier, "error": f"Structured fail: {e}"}; data_type = "error"
             elif data_type == "unstructured":
                 logger.debug(f"Running unstructured summarization: {identifier}")
                 doc_content = None; doc_type = 'unknown'
                 if preloaded_content is not None: doc_content = preloaded_content; doc_type = os.path.splitext(identifier)[1].lower().strip('.') if identifier and '.' in identifier else 'preloaded'
                 elif isinstance(data_input, str):
                     try: # Sync DataLoader calls
                         file_ext = os.path.splitext(identifier)[1].lower()
                         loader_func = None
                         if file_ext == ".pdf": loader_func = DataLoader.load_pdf
                         elif file_ext in [".txt", ".md", ".log", ".py", ".js", ".html", ".css"]: loader_func = DataLoader.load_text
                         elif file_ext == ".docx": loader_func = DataLoader.load_docx
                         else: raise ValueError(f"Unsupported extension '{file_ext}'.")
                         if loader_func: loaded_docs_list = loader_func(data_input)
                         else: loaded_docs_list = []
                         if not loaded_docs_list: raise ValueError("DataLoader empty.")
                         doc_content = "\n---\n".join([doc.page_content for doc in loaded_docs_list if hasattr(doc, 'page_content') and doc.page_content])
                         if not doc_content: raise ValueError("No text extracted.")
                         doc_type = file_ext.strip('.')
                     except Exception as load_err: logger.error(f"Load fail: {load_err}"); summary_result = {"identifier": identifier, "error": f"Load fail: {load_err}"}; data_type = "error"
                 else: summary_result = {"identifier": identifier or "unknown", "error": "Missing content source."}; data_type = "error"
                 # Call LLM summarizer (sync)
                 if data_type != "error" and doc_content:
                     try:
                         summary_result = self._generate_llm_summary_from_content(content=doc_content, identifier=identifier or "unknown", doc_type=doc_type)
                         if "error" in summary_result: data_type = "error"
                     except Exception as llm_err: logger.error(f"LLM call failed: {llm_err}"); summary_result = {"identifier": identifier or "unknown", "error": f"LLM fail: {llm_err}"}; data_type = "error"

        # --- Format Final Output ---
        if identifier and "identifier" not in summary_result: summary_result["identifier"] = identifier
        elif not identifier and "identifier" not in summary_result: summary_result["identifier"] = "unknown_source_final"
        if "source_type" not in summary_result and source_type != "unknown": summary_result["source_type"] = source_type
        if not isinstance(summary_result, dict): logger.critical(f"Internal Error: summary_result not dict!"); summary_result = {"identifier": identifier or "unknown_fmt", "error": "Internal format error.", "source_type": source_type}; data_type = "error"

        final_output = self._format_output_json(summary_result, data_type, source_type, target_db, target_table)
        logger.info(f"Summarization complete: '{final_output.get('metadata', {}).get('identifier', 'unknown_id')}'. ID: {final_output.get('id', 'no_id')}")
        return final_output