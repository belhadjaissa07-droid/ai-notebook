"""Step 3 — Answer generation: put the retrieved chunks in the prompt and stream the answer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.rag.shared.models import create_chat_model, message_text
from src.rag.shared.types import RetrievedChunk

NOT_ENOUGH_INFO = "I don't have enough information in the provided documents to answer this question."

GROUNDED_SYSTEM_PROMPT = """You are a friendly Study Tutor for university students.
Answer from the course excerpts you are given, and explain step by step when it helps.
Answer in the language of the student's question."""


def to_langchain_messages(history: Sequence[Mapping[str, Any]]) -> list[BaseMessage]:
    message_types = {"user": HumanMessage, "assistant": AIMessage}
    return [message_types[turn["role"]](content=turn["content"]) for turn in history]


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{number}] {chunk.text}" for number, chunk in enumerate(chunks, start=1))


def build_grounded_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """The augmented prompt = the question + the retrieved excerpts."""
    return f"""Using only the course excerpts below, answer the question.

Question: {question}

Course excerpts:
{format_context(chunks)}

Instructions:
- Use only the information in the excerpts above.
- When you rely on an excerpt, mention its number, like [1].
- If the answer is not in the excerpts, reply: "{NOT_ENOUGH_INFO}"
"""


def stream_grounded_answer(
    history: Sequence[Mapping[str, Any]],
    question: str,
    chunks: Sequence[RetrievedChunk],
    temperature: float,
) -> Iterator[str]:
    messages = [
        SystemMessage(content=GROUNDED_SYSTEM_PROMPT),
        *to_langchain_messages(history),
        HumanMessage(content=build_grounded_prompt(question, chunks)),
    ]
    for chunk in create_chat_model(temperature).stream(messages):
        text = message_text(chunk.content)
        if text:
            yield text
