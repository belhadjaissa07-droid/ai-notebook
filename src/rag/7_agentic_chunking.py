"""Step 7 — Agentic chunking: ask the LLM to mark each natural topic boundary."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.shared.models import create_chat_model, message_text

SPLIT_MARKER = "<<<SPLIT>>>"
WINDOW_CHARS = 4000

AGENTIC_PROMPT = f"""You are an expert at splitting text for a RAG system.
Split the text below into logical, coherent chunks.

Rules:
- Keep together the information that deals with the same topic.
- Keep each chunk concise (about 250 to 350 characters at most).
- Insert exactly the marker {SPLIT_MARKER} between two chunks.
- Do not change the original text and do not add any comment.

Text:
"""


def split_agentically(documents: list[Document]) -> list[Document]:
    """Long documents are sent window by window to stay within the model's output limit."""
    llm = create_chat_model(temperature=0.0)
    window_splitter = RecursiveCharacterTextSplitter(chunk_size=WINDOW_CHARS, chunk_overlap=0)
    chunks: list[Document] = []
    for document in documents:
        for window in window_splitter.split_text(document.page_content):
            marked_text = message_text(llm.invoke(AGENTIC_PROMPT + window).content)
            for piece in marked_text.split(SPLIT_MARKER):
                if piece.strip():
                    chunks.append(Document(page_content=piece.strip(), metadata=document.metadata))
    return chunks
