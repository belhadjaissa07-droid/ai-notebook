"""Step 1 — Ingestion pipeline: documents -> chunks -> embeddings -> Chroma."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from importlib import import_module

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from src.rag.shared.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.rag.storage.store import delete_source, get_vector_store

# File names start with a digit, like the workshop scripts, so `import` cannot reach them.
recursive_splitter = import_module("src.rag.5_recursive_character_text_splitter")
agentic_chunking = import_module("src.rag.7_agentic_chunking")


def split_by_character(documents: list[Document]) -> list[Document]:
    """Cut on blank lines only, then merge paragraphs up to CHUNK_SIZE."""
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


CHUNKING_STRATEGIES: dict[str, Callable[[list[Document]], list[Document]]] = {
    "Character": split_by_character,
    "Recursive": recursive_splitter.split_recursively,
    "Agentic (LLM)": agentic_chunking.split_agentically,
}


def number_chunks(chunks: list[Document]) -> list[Document]:
    """Give each chunk its position inside its source (1, 2, 3...) for citations."""
    seen: Counter[str] = Counter()
    for chunk in chunks:
        source = chunk.metadata["source"]
        seen[source] += 1
        chunk.metadata["chunk"] = seen[source]
    return chunks


def ingest_documents(documents: list[Document], strategy: str) -> int:
    """Index documents with the chosen chunking strategy; returns the chunk count.

    Re-ingesting a document replaces its previous chunks.
    """
    chunks = number_chunks(CHUNKING_STRATEGIES[strategy](documents))
    for source in {document.metadata["source"] for document in documents}:
        delete_source(source)
    if chunks:
        get_vector_store().add_documents(chunks)
    return len(chunks)
