"""Step 1 — Ingestion pipeline: load documents -> split into chunks -> embed and store in Chroma."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from src.rag.shared.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
)
from src.rag.shared.models import create_embeddings

# File names start with a digit, like the workshop scripts, so `import` cannot reach them.
recursive_splitter = import_module("src.rag.5_recursive_character_text_splitter")
agentic_chunking = import_module("src.rag.7_agentic_chunking")

DB_DIR = str(CHROMA_DIR)


class UploadedFile(Protocol):
    """What the UI's file uploader gives us."""

    name: str

    def getvalue(self) -> bytes: ...


def load_documents(uploaded_files: Sequence[UploadedFile]) -> list[Document]:
    """Read the .txt files uploaded in the UI as LangChain documents."""
    documents = []
    for uploaded_file in uploaded_files:
        try:
            text = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{uploaded_file.name} is not a UTF-8 text file.") from error
        documents.append(Document(page_content=text, metadata={"source": uploaded_file.name}))
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split the documents into chunks with overlap."""
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks: list[Document], persist_directory: str = DB_DIR) -> Chroma:
    """Generate the Gemini embeddings and save them in ChromaDB."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=create_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )


CHUNKING_STRATEGIES: dict[str, Callable[[list[Document]], list[Document]]] = {
    "Character": split_documents,
    "Recursive": recursive_splitter.split_documents_recursively,
    "Agentic (LLM)": agentic_chunking.split_documents_agentically,
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
    """Chunk the documents with the chosen strategy and store them in Chroma; returns the chunk count."""
    chunks = number_chunks(CHUNKING_STRATEGIES[strategy](documents))
    if chunks:
        create_vector_store(chunks, persist_directory=DB_DIR)
    return len(chunks)
