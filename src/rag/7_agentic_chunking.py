"""Step 7 — Agentic chunking: ask the LLM to mark each natural topic boundary."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.shared.messages import message_text
from src.rag.shared.models import create_chat_model

SPLIT_MARKER = "<<<SPLIT>>>"
WINDOW_CHARS = 4000

AGENTIC_PROMPT = f"""You are an expert at splitting text for a RAG system.
Split the text below into logical, coherent chunks.

Rules:
- Group together the information that deals with the same mathematical topic.
- Keep each chunk concise (about 250 to 350 characters at most).
- Insert exactly the marker {SPLIT_MARKER} between two chunks.
- Do not change the original content and do not add any comment.

Text:
"""


def split_documents_agentically(documents: list[Document]) -> list[Document]:
    """Long documents are sent window by window to stay within the model's output limit."""
    # TODO (workshop):
    # 1. llm = create_chat_model(temperature=0.0)
    # 2. Send long documents window by window: cut each one with a
    #    RecursiveCharacterTextSplitter(chunk_size=WINDOW_CHARS, chunk_overlap=0)
    # 3. For each window: llm.invoke(AGENTIC_PROMPT + window), read the text with
    #    message_text(response.content)
    # 4. Split that text on SPLIT_MARKER; every non-empty piece becomes
    #    Document(page_content=piece.strip(), metadata=document.metadata)
    raise NotImplementedError("Step 7: implement split_documents_agentically")
