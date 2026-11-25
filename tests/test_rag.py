from __future__ import annotations

from importlib import import_module
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.rag import (
    CHUNKING_STRATEGIES,
    ask_question,
    clear_store,
    count_chunks,
    delete_source,
    format_context,
    ingest_documents,
    list_sources,
    load_documents,
    to_chat_history,
)

agentic_chunking = import_module("src.rag.7_agentic_chunking")
answer_generation = import_module("src.rag.3_answer_generation")
history_aware = import_module("src.rag.4_history_aware_generation")
ingestion = import_module("src.rag.1_ingestion_pipeline")
retrieval_pipeline = import_module("src.rag.2_retrieval_pipeline")

SPLIT_MARKER = agentic_chunking.SPLIT_MARKER
NOT_ENOUGH_INFO = answer_generation.NOT_ENOUGH_INFO
build_augmented_prompt = answer_generation.build_augmented_prompt
build_standalone_question = history_aware.build_standalone_question
retrieve = retrieval_pipeline.retrieve

LONG_TEXT = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(12))


def make_document(source: str, text: str = LONG_TEXT) -> Document:
    return Document(page_content=text, metadata={"source": source})


class FakeUpload:
    """Looks like the UI's uploaded file: a name and getvalue()."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_sample_course_loads_four_chapters_named_by_file():
    sample_files = [FakeUpload(path.name, path.read_bytes()) for path in Path("data/sample_docs").glob("*.txt")]
    documents = load_documents(sample_files)
    assert sorted(d.metadata["source"] for d in documents) == [
        "matrices_01_definitions_operations.txt",
        "matrices_02_determinant_inverse.txt",
        "matrices_03_systemes_lineaires.txt",
        "matrices_04_valeurs_propres.txt",
    ]


def test_load_documents_uses_the_uploaded_file_name_as_source():
    [document] = load_documents([FakeUpload("notes.txt", "héllo".encode())])
    assert document.metadata["source"] == "notes.txt"
    assert document.page_content == "héllo"


def test_load_documents_rejects_a_file_that_is_not_utf8():
    try:
        load_documents([FakeUpload("binary.txt", b"\xff\xfe\x00")])
    except ValueError as error:
        assert "binary.txt" in str(error)
        return
    raise AssertionError("expected ValueError")


def test_split_documents_cuts_on_blank_lines_within_the_size_limit():
    chunks = ingestion.split_documents([make_document("a.txt")], chunk_size=700, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 700 for chunk in chunks)


def test_recursive_chunks_respect_the_size_limit():
    chunks = CHUNKING_STRATEGIES["Recursive"]([make_document("a.txt", "x " * 2000)])
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 800 for chunk in chunks)


def test_agentic_chunking_splits_on_the_marker(fake_chat):
    fake_chat[:] = [f"First idea.{SPLIT_MARKER}Second idea.{SPLIT_MARKER}  "]
    chunks = CHUNKING_STRATEGIES["Agentic (LLM)"]([make_document("a.txt", "short text")])
    assert [c.page_content for c in chunks] == ["First idea.", "Second idea."]
    assert chunks[0].metadata["source"] == "a.txt"


def test_ingest_numbers_chunks_per_source():
    documents = [make_document("a.txt"), make_document("b.txt")]
    added = ingest_documents(documents, "Recursive")
    assert count_chunks() == added
    assert list_sources() == ["a.txt", "b.txt"]

    numbers = sorted(c.metadata["chunk"] for c in retrieve("Paragraph", top_k=50) if c.metadata["source"] == "a.txt")
    assert numbers == list(range(1, len(numbers) + 1))


def test_delete_and_clear():
    ingest_documents([make_document("a.txt"), make_document("b.txt")], "Recursive")
    delete_source("a.txt")
    assert list_sources() == ["b.txt"]
    clear_store()
    assert list_sources() == []
    assert count_chunks() == 0


def test_inventory_needs_no_embeddings(monkeypatch):
    def fail():
        raise AssertionError("embeddings must not be created")

    monkeypatch.setattr("src.rag.storage.store.create_embeddings", fail)
    assert list_sources() == []
    assert count_chunks() == 0


def test_retrieve_returns_the_closest_chunk_first():
    ingest_documents([make_document("a.txt", "Eigenvalues of a matrix.")], "Character")
    ingest_documents([make_document("b.txt", "Cooking pasta takes ten minutes.")], "Character")
    chunks = retrieve("Eigenvalues of a matrix.", top_k=2)
    assert [c.metadata["source"] for c in chunks] == ["a.txt", "b.txt"]


def test_augmented_prompt_contains_question_excerpts_and_fallback():
    docs = [make_document("a.txt", "Alpha."), make_document("b.txt", "Beta.")]
    prompt = build_augmented_prompt("What is alpha?", docs)
    assert "What is alpha?" in prompt
    assert "- Alpha." in prompt and "- Beta." in prompt
    assert NOT_ENOUGH_INFO in prompt
    assert format_context(docs) in prompt


def test_chat_history_keeps_the_order_and_roles():
    history = to_chat_history(
        [
            {"role": "user", "content": "Hi", "chunks": []},
            {"role": "assistant", "content": "Hello", "chunks": []},
        ]
    )
    assert [type(m) for m in history] == [HumanMessage, AIMessage]
    assert [m.content for m in history] == ["Hi", "Hello"]


def test_first_question_is_not_rewritten():
    assert build_standalone_question(None, [], "What is a matrix?") == "What is a matrix?"


def test_follow_up_is_rewritten_with_history(fake_chat):
    fake_chat[:] = ["What are the eigenvectors of Q?"]
    history = [HumanMessage(content="What are the eigenvalues of Q?"), AIMessage(content="They are 1 and 3.")]
    model = history_aware.create_chat_model()
    assert build_standalone_question(model, history, "and its eigenvectors?") == (
        "What are the eigenvectors of Q?"
    )


def test_ask_question_streams_a_reply_with_its_sources(fake_chat):
    fake_chat[:] = ["Alpha is the first letter."]
    ingest_documents([make_document("a.txt", "Alpha is the first letter.")], "Character")
    answer = ask_question([], "What is alpha?", top_k=1)
    assert answer.search_question == "What is alpha?"
    assert [c.metadata["source"] for c in answer.chunks] == ["a.txt"]
    assert "".join(answer.tokens) == "Alpha is the first letter."
