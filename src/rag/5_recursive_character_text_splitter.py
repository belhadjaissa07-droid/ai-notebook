"""Step 5 — Recursive chunking: try paragraphs, then lines, sentences and words."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.shared.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_documents_recursively(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Try several separators in order until every chunk fits."""
    # TODO (workshop):
    # 1. Create a RecursiveCharacterTextSplitter with
    #    separators: paragraph ("\n\n"), line ("\n"), sentence (". "), word (" "), character ("")
    #    chunk_size=chunk_size, chunk_overlap=chunk_overlap
    # 2. Return splitter.split_documents(documents)
    raise NotImplementedError("Step 5: implement split_documents_recursively")
