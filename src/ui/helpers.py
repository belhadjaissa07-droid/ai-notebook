"""Pure helpers for the Streamlit UI (no Streamlit import, easy to unit test)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence

from langchain_core.documents import Document


def citation_chip_label(index: int, chunk: Document) -> str:
    return f"[{index}] {chunk.metadata['source']} · chunk {chunk.metadata['chunk']}"


def answer_summary(
    chunks: Sequence[Document],
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
    sources = ", ".join(sorted({chunk.metadata["source"] for chunk in chunks}))
    parts.append(f"Read {len(chunks)} excerpt{'s' if len(chunks) != 1 else ''} from {sources}.")
    return " ".join(parts)


def stream_once_ready(tokens: Iterable[str], on_first_token: Callable[[], None]) -> Iterator[str]:
    """Yield the tokens, calling `on_first_token` just before the first one (or at the end if none)."""
    tokens = iter(tokens)
    first = next(tokens, None)
    on_first_token()
    if first is not None:
        yield first
        yield from tokens


def starter_prompts(sources: Sequence[str]) -> list[tuple[str, str]]:
    """(chip label, prompt text) suggestions built from the first indexed document."""
    if not sources:
        return []
    document = sources[0]
    return [
        ("Summarize it", f"Summarize {document} in a few bullet points."),
        ("Key terms", f"List the key terms in {document} with one-line definitions."),
    ]
