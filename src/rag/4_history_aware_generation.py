"""Step 4 — History-aware generation: rewrite follow-up questions, then retrieve and answer."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib import import_module

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.rag.shared.config import DEFAULT_TOP_K
from src.rag.shared.messages import message_text
from src.rag.shared.models import create_chat_model

# File names start with a digit, like the workshop scripts, so `import` cannot reach them.
retrieval_pipeline = import_module("src.rag.2_retrieval_pipeline")
answer_generation = import_module("src.rag.3_answer_generation")

REWRITE_SYSTEM_PROMPT = """From the conversation history, rewrite the new question so it is
standalone and clear for a document search.
Return only the rewritten question."""


@dataclass(frozen=True)
class RagAnswer:
    search_question: str
    chunks: list[Document]
    tokens: Iterator[str]


def build_standalone_question(model, chat_history: Sequence[BaseMessage], user_question: str) -> str:
    """Rewrite a follow-up like "and its eigenvectors?" so the search understands it."""
    if not chat_history:
        return user_question

    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        *chat_history,
        HumanMessage(content=f"New question: {user_question}"),
    ]
    return message_text(model.invoke(messages).content).strip() or user_question


def ask_question(
    chat_history: Sequence[BaseMessage],
    user_question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = 0.3,
) -> RagAnswer:
    """One full conversational RAG turn. `chat_history` holds the previous turns only."""
    search_question = build_standalone_question(create_chat_model(), chat_history, user_question)
    docs = retrieval_pipeline.retrieve(search_question, top_k)
    tokens = answer_generation.generate_answer(user_question, docs, temperature, chat_history)
    return RagAnswer(search_question=search_question, chunks=docs, tokens=tokens)
