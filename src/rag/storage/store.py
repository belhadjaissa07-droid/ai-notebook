"""Vector store: the Chroma database on disk (cosine), plus list, count and delete."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from src.rag.shared.config import CHROMA_DIR, COLLECTION_NAME
from src.rag.shared.models import create_embeddings


def _open_store(embedding_function: Embeddings | None) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_vector_store() -> Chroma:
    """Store that can embed text: use it to add chunks and to search."""
    return _open_store(create_embeddings())


def list_sources() -> list[str]:
    """Names of the indexed documents. Needs no embedding, so no API key."""
    metadatas = _open_store(None).get(include=["metadatas"])["metadatas"]
    return sorted({metadata["source"] for metadata in metadatas})


def count_chunks() -> int:
    return len(_open_store(None).get(include=[])["ids"])


def delete_source(source: str) -> None:
    _open_store(None).delete(where={"source": source})


def clear_store() -> None:
    _open_store(None).delete_collection()
