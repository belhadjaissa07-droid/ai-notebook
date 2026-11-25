from __future__ import annotations

from importlib import import_module

from langchain_core.documents import Document

from src.rag import (
    CHUNKING_STRATEGIES,
    SAMPLE_DOCS_DIR,
    answer_question,
    clear_store,
    count_chunks,
    delete_source,
    format_context,
    ingest_documents,
    list_sources,
    load_text_folder,
    load_text_upload,
)

agentic_chunking = import_module("src.rag.7_agentic_chunking")
answer_generation = import_module("src.rag.3_answer_generation")
history_aware = import_module("src.rag.4_history_aware_generation")
retrieval_pipeline = import_module("src.rag.2_retrieval_pipeline")

SPLIT_MARKER = agentic_chunking.SPLIT_MARKER
NOT_ENOUGH_INFO = answer_generation.NOT_ENOUGH_INFO
build_grounded_prompt = answer_generation.build_grounded_prompt
rewrite_as_standalone_question = history_aware.rewrite_as_standalone_question
retrieve = retrieval_pipeline.retrieve

LONG_TEXT = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(12))


def make_document(source: str, text: str = LONG_TEXT) -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_sample_course_loads_four_chapters_named_by_file():
    documents = load_text_folder(SAMPLE_DOCS_DIR)
    assert [d.metadata["source"] for d in documents] == [
        "matrices_01_definitions_operations.txt",
        "matrices_02_determinant_inverse.txt",
        "matrices_03_systemes_lineaires.txt",
        "matrices_04_valeurs_propres.txt",
    ]


def test_upload_uses_only_the_file_name():
    [document] = load_text_upload("../secret/notes.txt", "héllo".encode())
    assert document.metadata["source"] == "notes.txt"
    assert document.page_content == "héllo"


def test_chunks_respect_the_size_limit_with_recursive_splitting():
    chunks = CHUNKING_STRATEGIES["Recursive"]([make_document("a.txt", "x " * 2000)])
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 800 for chunk in chunks)


def test_agentic_chunking_splits_on_the_marker(fake_chat):
    fake_chat[:] = [f"First idea.{SPLIT_MARKER}Second idea.{SPLIT_MARKER}  "]
    chunks = CHUNKING_STRATEGIES["Agentic (LLM)"]([make_document("a.txt", "short text")])
    assert [c.page_content for c in chunks] == ["First idea.", "Second idea."]
    assert chunks[0].metadata["source"] == "a.txt"


def test_ingest_numbers_chunks_per_source_and_replaces_on_reingest():
    documents = [make_document("a.txt"), make_document("b.txt")]
    added = ingest_documents(documents, "Recursive")
    assert count_chunks() == added
    assert list_sources() == ["a.txt", "b.txt"]

    ingest_documents([make_document("a.txt")], "Recursive")
    assert count_chunks() == added

    numbers = sorted(c.number for c in retrieve("Paragraph", top_k=50) if c.source == "a.txt")
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


def test_retrieve_returns_closest_chunk_first():
    ingest_documents([make_document("a.txt", "Eigenvalues of a matrix.")], "Character")
    ingest_documents([make_document("b.txt", "Cooking pasta takes ten minutes.")], "Character")
    chunks = retrieve("Eigenvalues of a matrix.", top_k=2)
    assert [c.source for c in chunks] == ["a.txt", "b.txt"]
    assert chunks[0].distance < chunks[1].distance


def test_grounded_prompt_numbers_the_excerpts():
    ingest_documents([make_document("a.txt", "Alpha."), make_document("b.txt", "Beta.")], "Character")
    chunks = retrieve("Alpha.", top_k=2)
    prompt = build_grounded_prompt("What is alpha?", chunks)
    assert "[1] " in prompt and "[2] " in prompt
    assert "What is alpha?" in prompt
    assert NOT_ENOUGH_INFO in prompt
    assert format_context(chunks) in prompt


def test_first_question_is_not_rewritten(fake_chat):
    assert rewrite_as_standalone_question([], "What is a matrix?") == "What is a matrix?"


def test_follow_up_is_rewritten_with_history(fake_chat):
    fake_chat[:] = ["What are the eigenvectors of Q?"]
    history = [
        {"role": "user", "content": "What are the eigenvalues of Q?"},
        {"role": "assistant", "content": "They are 1 and 3.", "chunks": []},
    ]
    assert rewrite_as_standalone_question(history, "and its eigenvectors?") == (
        "What are the eigenvectors of Q?"
    )


def test_answer_question_streams_a_reply_with_its_sources(fake_chat):
    fake_chat[:] = ["Alpha is the first letter [1]."]
    ingest_documents([make_document("a.txt", "Alpha is the first letter.")], "Character")
    answer = answer_question([], "What is alpha?", top_k=1)
    assert answer.search_question == "What is alpha?"
    assert [c.source for c in answer.chunks] == ["a.txt"]
    assert "".join(answer.tokens) == "Alpha is the first letter [1]."
