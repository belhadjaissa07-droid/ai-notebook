from __future__ import annotations

from langchain_core.documents import Document

from src.ui.helpers import answer_summary, citation_chip_label, starter_prompts, stream_once_ready


def make_chunk(source: str, number: int) -> Document:
    return Document(page_content="Some text", metadata={"source": source, "chunk": number})


CHUNK = make_chunk("a.txt", 2)


def test_chip_label_shows_source_and_chunk():
    assert citation_chip_label(1, CHUNK) == "[1] a.txt · chunk 2"


def test_summary_without_rag():
    summary = answer_summary([], question="Hi", search_question=None)
    assert "general knowledge" in summary


def test_summary_mentions_rewritten_follow_up():
    summary = answer_summary([CHUNK], question="and its trace?", search_question="What is the trace of A?")
    assert "Rewrote your follow-up" in summary
    assert "1 excerpt from a.txt" in summary


def test_summary_pluralises_and_lists_each_source_once():
    summary = answer_summary([CHUNK, CHUNK, make_chunk("b.txt", 1)], question="Q", search_question="Q")
    assert "3 excerpts from a.txt, b.txt" in summary


def test_starters_need_a_document():
    assert starter_prompts([]) == []
    assert all("a.txt" in text for _, text in starter_prompts(["a.txt", "b.txt"]))


def test_waiting_message_goes_away_right_before_the_first_token():
    events = []

    def tokens():
        events.append("model started")
        yield "Hello"
        yield " world"

    stream = stream_once_ready(tokens(), lambda: events.append("hidden"))
    assert events == []
    assert list(stream) == ["Hello", " world"]
    assert events == ["model started", "hidden"]


def test_waiting_message_goes_away_even_when_the_model_returns_nothing():
    events = []
    assert list(stream_once_ready(iter([]), lambda: events.append("hidden"))) == []
    assert events == ["hidden"]
