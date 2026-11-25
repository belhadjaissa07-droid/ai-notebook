"""Session 2 — RAG: load -> chunk -> embed -> store -> retrieve -> generate.

The numbered files are the workshop steps. Their names start with a digit, so they are
loaded with `import_module`; the rest of the app only uses the names exported here.
"""

from importlib import import_module

from src.rag.shared.config import DEFAULT_TOP_K
from src.rag.shared.messages import to_chat_history
from src.rag.storage.store import clear_store, count_chunks, delete_source, list_sources

_ingestion = import_module("src.rag.1_ingestion_pipeline")
_answer_generation = import_module("src.rag.3_answer_generation")
_history_aware = import_module("src.rag.4_history_aware_generation")

CHUNKING_STRATEGIES = _ingestion.CHUNKING_STRATEGIES
load_documents = _ingestion.load_documents
ingest_documents = _ingestion.ingest_documents
format_context = _answer_generation.format_context
RagAnswer = _history_aware.RagAnswer
ask_question = _history_aware.ask_question

__all__ = [
    "CHUNKING_STRATEGIES",
    "DEFAULT_TOP_K",
    "RagAnswer",
    "ask_question",
    "clear_store",
    "count_chunks",
    "delete_source",
    "format_context",
    "ingest_documents",
    "list_sources",
    "load_documents",
    "to_chat_history",
]
