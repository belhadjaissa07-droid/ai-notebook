from __future__ import annotations

import itertools
from importlib import import_module

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.rag.storage import store

ingestion = import_module("src.rag.1_ingestion_pipeline")
agentic_chunking = import_module("src.rag.7_agentic_chunking")
answer_generation = import_module("src.rag.3_answer_generation")
history_aware = import_module("src.rag.4_history_aware_generation")


@pytest.fixture(autouse=True)
def isolated_rag(tmp_path, monkeypatch):
    """Temporary Chroma folder + fake embeddings, so tests never touch the API."""
    chroma_dir = tmp_path / "chroma"
    fake_embeddings = lambda: DeterministicFakeEmbedding(size=32)  # noqa: E731
    monkeypatch.setattr(store, "CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(store, "create_embeddings", fake_embeddings)
    monkeypatch.setattr(ingestion, "DB_DIR", str(chroma_dir))
    monkeypatch.setattr(ingestion, "create_embeddings", fake_embeddings)


@pytest.fixture
def fake_chat(monkeypatch):
    """Replace the Gemini chat model; returns the list of replies to cycle through."""
    replies = ["Fake reply."]

    def factory(temperature: float = 0.0):
        return GenericFakeChatModel(messages=itertools.cycle(AIMessage(content=r) for r in replies))

    for module in (agentic_chunking, answer_generation, history_aware):
        monkeypatch.setattr(module, "create_chat_model", factory)
    return replies
