"""Session 2 settings: models, folders and RAG numbers in one place."""

from __future__ import annotations

import os
from pathlib import Path

from src.llm import DEFAULT_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_DOCS_DIR = PROJECT_ROOT / "data" / "sample_docs"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "course_docs"

CHAT_MODEL = DEFAULT_MODEL
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 3
