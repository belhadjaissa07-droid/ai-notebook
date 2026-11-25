"""Step 2 — Retrieval pipeline: find the chunks closest in meaning to a question."""

from __future__ import annotations

from langchain_core.documents import Document

from src.rag.shared.config import DEFAULT_TOP_K
from src.rag.storage.store import get_vector_store


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[Document]:
    """Search ChromaDB by cosine similarity and return the k closest chunks."""
    # TODO (workshop):
    # 1. Connect to the vector database: db = get_vector_store()
    # 2. Create a retriever: db.as_retriever(search_kwargs={"k": top_k})
    # 3. Return retriever.invoke(query)
    raise NotImplementedError("Step 2: implement retrieve")
