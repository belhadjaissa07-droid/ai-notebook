"""Session 2 — RAG: load -> chunk -> embed -> store -> retrieve -> generate."""

from importlib import import_module

from src.rag.shared.config import DEFAULT_TOP_K, SAMPLE_DOCS_DIR
from src.rag.storage.loader import load_text_folder, load_text_upload
from src.rag.storage.store import clear_store, count_chunks, delete_source, list_sources
from src.rag.shared.types import RetrievedChunk

_ingestion = import_module("src.rag.1_ingestion_pipeline")
_generation = import_module("src.rag.3_answer_generation")
_history_aware = import_module("src.rag.4_history_aware_generation")

CHUNKING_STRATEGIES = _ingestion.CHUNKING_STRATEGIES
ingest_documents = _ingestion.ingest_documents
format_context = _generation.format_context
RagAnswer = _history_aware.RagAnswer
answer_question = _history_aware.answer_question

__all__ = [
    "CHUNKING_STRATEGIES",
    "DEFAULT_TOP_K",
    "SAMPLE_DOCS_DIR",
    "RagAnswer",
    "RetrievedChunk",
    "answer_question",
    "clear_store",
    "count_chunks",
    "delete_source",
    "format_context",
    "ingest_documents",
    "list_sources",
    "load_text_folder",
    "load_text_upload",
]
