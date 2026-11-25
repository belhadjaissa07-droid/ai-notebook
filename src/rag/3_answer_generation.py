"""Step 3 — Answer generation: put the retrieved chunks in the prompt and let Gemini answer."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.rag.shared.messages import message_text
from src.rag.shared.models import create_chat_model

NOT_ENOUGH_INFO = "I don't have enough information in the provided documents to answer this question."

SYSTEM_PROMPT = """You are a Study Tutor for university students.
Rely on the course excerpts and on the conversation history.
Answer in the language of the student's question."""


def format_context(docs: Sequence[Document]) -> str:
    return "\n\n".join(f"- {doc.page_content}" for doc in docs)


def build_augmented_prompt(question: str, docs: Sequence[Document]) -> str:
    """The augmented prompt = the question + the retrieved excerpts + instructions."""
    return f"""Using only the following course excerpts, answer the question clearly: {question}

Course excerpts:
{format_context(docs)}

Instructions:
- Use only the information present in the excerpts above.
- If the information is not there, reply: "{NOT_ENOUGH_INFO}"
"""


def generate_answer(
    question: str,
    docs: Sequence[Document],
    temperature: float = 0.3,
    chat_history: Sequence[BaseMessage] = (),
) -> Iterator[str]:
    """Ask Gemini and stream the answer piece by piece."""
    model = create_chat_model(temperature)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *chat_history,
        HumanMessage(content=build_augmented_prompt(question, docs)),
    ]
    for piece in model.stream(messages):
        text = message_text(piece.content)
        if text:
            yield text
