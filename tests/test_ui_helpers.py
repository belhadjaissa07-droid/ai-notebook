from __future__ import annotations

from src.rag import RetrievedChunk
from src.ui.helpers import answer_summary, citation_chip_label, starter_prompts

CHUNK = RetrievedChunk(text="Some text", source="a.txt", number=2, distance=0.12)


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
    other = RetrievedChunk(text="x", source="b.txt", number=1, distance=0.3)
    summary = answer_summary([CHUNK, CHUNK, other], question="Q", search_question="Q")
    assert "3 excerpts from a.txt, b.txt" in summary


def test_starters_need_a_document():
    assert starter_prompts([]) == []
    assert all("a.txt" in text for _, text in starter_prompts(["a.txt", "b.txt"]))
