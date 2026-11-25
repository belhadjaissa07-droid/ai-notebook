"""LangChain wrappers around Gemini: one for embeddings, one for chat."""

from __future__ import annotations

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.llm import get_api_key
from src.rag.shared.config import CHAT_MODEL, EMBEDDING_MODEL


def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=get_api_key())


def create_chat_model(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=get_api_key(),
        temperature=temperature,
    )


def message_text(content: Any) -> str:
    """Gemini returns either a string or a list of content blocks; keep the text."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))
