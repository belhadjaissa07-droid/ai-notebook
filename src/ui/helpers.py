"""Pure helpers for the Streamlit UI (no Streamlit import, easy to unit test)."""

from __future__ import annotations

from collections.abc import Sequence

from src.rag.shared.types import RetrievedChunk


def citation_chip_label(index: int, chunk: RetrievedChunk) -> str:
    return f"[{index}] {chunk.source} · chunk {chunk.number}"


def answer_summary(
    chunks: Sequence[RetrievedChunk],
    *,
    question: str,
    search_question: str | None,
) -> str:
    """Plain-language account of how an answer was produced."""
    if search_question is None:
        return "Answered from the model's general knowledge. Your documents were not read."
    parts = []
    if search_question != question:
        parts.append(f"Rewrote your follow-up as “{search_question}”.")
    sources = ", ".join(sorted({chunk.source for chunk in chunks}))
    parts.append(f"Read {len(chunks)} excerpt{'s' if len(chunks) != 1 else ''} from {sources}.")
    return " ".join(parts)


def starter_prompts(sources: Sequence[str]) -> list[tuple[str, str]]:
    """(chip label, prompt text) suggestions built from the first indexed document."""
    if not sources:
        return []
    document = sources[0]
    return [
        ("Summarize it", f"Summarize {document} in a few bullet points."),
        ("Key terms", f"List the key terms in {document} with one-line definitions."),
    ]
