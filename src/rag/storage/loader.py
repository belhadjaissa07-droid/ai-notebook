"""Step 1a — Load documents: turn files into LangChain `Document` objects."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


def load_text_folder(folder: Path) -> list[Document]:
    """Load every .txt file of a folder (used for the sample course)."""
    loader = DirectoryLoader(
        path=str(folder),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    for document in documents:
        document.metadata["source"] = Path(document.metadata["source"]).name
    return sorted(documents, key=lambda document: document.metadata["source"])


def load_text_upload(filename: str, data: bytes) -> list[Document]:
    """Wrap an uploaded .txt file; raises UnicodeDecodeError if it is not UTF-8."""
    return [Document(page_content=data.decode("utf-8"), metadata={"source": Path(filename).name})]
