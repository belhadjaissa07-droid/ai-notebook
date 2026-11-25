"""Step 4 — History-aware generation: rewrite follow-up questions, then retrieve and answer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.rag.shared.config import DEFAULT_TOP_K
from src.rag.shared.models import create_chat_model, message_text
from src.rag.shared.types import RetrievedChunk

# File names start with a digit, like the workshop scripts, so `import` cannot reach them.
retrieval_pipeline = import_module("src.rag.2_retrieval_pipeline")
answer_generation = import_module("src.rag.3_answer_generation")

REWRITE_SYSTEM_PROMPT = """Using the conversation so far, rewrite the new question so it
makes sense on its own and works for a document search.
Return only the rewritten question."""


@dataclass(frozen=True)
class RagAnswer:
    search_question: str
    chunks: list[RetrievedChunk]
    tokens: Iterator[str]


def rewrite_as_standalone_question(history: Sequence[Mapping[str, Any]], question: str) -> str:
    """Turn a follow-up like "and its eigenvectors?" into a full question."""
    if not history:
        return question
    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        *answer_generation.to_langchain_messages(history),
        HumanMessage(content=f"New question: {question}"),
    ]
    rewritten = message_text(create_chat_model().invoke(messages).content).strip()
    return rewritten or question


def answer_question(
    history: Sequence[Mapping[str, Any]],
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = 0.3,
) -> RagAnswer:
    """Full RAG turn: rewrite, retrieve, then stream a grounded answer.

    `history` holds the previous turns only (not the new question).
    """
    search_question = rewrite_as_standalone_question(history, question)
    chunks = retrieval_pipeline.retrieve(search_question, top_k)
    return RagAnswer(
        search_question=search_question,
        chunks=chunks,
        tokens=answer_generation.stream_grounded_answer(history, question, chunks, temperature),
    )
