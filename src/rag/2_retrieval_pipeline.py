"""Step 2 — Retrieval pipeline: find the chunks closest in meaning to a question."""

from __future__ import annotations

from src.rag.shared.config import DEFAULT_TOP_K
from src.rag.storage.store import get_vector_store
from src.rag.shared.types import RetrievedChunk


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """Embed the query and return the `top_k` nearest chunks (smaller distance = closer)."""
    results = get_vector_store().similarity_search_with_score(query, k=top_k)
    return [
        RetrievedChunk(
            text=document.page_content,
            source=document.metadata["source"],
            number=document.metadata["chunk"],
            distance=round(distance, 3),
        )
        for document, distance in results
    ]
