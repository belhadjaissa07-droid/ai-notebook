"""AI Notebook — Session 2: the Study Tutor answers from your documents (RAG)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.ui.components import render_answer_extras, render_preview_pane, render_trace_panel
from src.llm import DEFAULT_MODEL, chat_stream
from src.rag import (
    CHUNKING_STRATEGIES,
    DEFAULT_TOP_K,
    ask_question,
    clear_store,
    count_chunks,
    delete_source,
    ingest_documents,
    list_sources,
    load_documents,
    to_chat_history,
)
from src.rag.shared.config import EMBEDDING_MODEL
from src.ui.theme import apply_theme
from src.ui.helpers import starter_prompts, stream_once_ready

st.set_page_config(page_title="AI Notebook — Study Tutor", layout="wide")
apply_theme()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("selected_chunk", None)
st.session_state.setdefault("uploader_version", 0)
st.session_state.setdefault("flashes", [])


def flash(level: str, message: str) -> None:
    """Queue a message (st.success / st.error / ...) to show after the next rerun."""
    st.session_state.flashes.append((level, message))


def reset_library_widgets() -> None:
    st.session_state.selected_chunk = None
    # A new key empties the uploader, so a removed file is not indexed again.
    st.session_state.uploader_version += 1


def remove_document(source: str) -> None:
    delete_source(source)
    reset_library_widgets()


def clear_all_documents() -> None:
    clear_store()
    reset_library_widgets()


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.selected_chunk = None


def queue_prompt(text: str) -> None:
    st.session_state.queued_prompt = text


def index_documents(documents: list, label: str, strategy: str) -> None:
    try:
        with st.spinner(f"Indexing {label} ({strategy} chunking)…"):
            chunk_count = ingest_documents(documents, strategy)
    except ValueError as exc:
        flash("error", str(exc))
    except Exception as exc:  # noqa: BLE001
        flash("error", f"Indexing {label} failed: {exc}")
    else:
        if chunk_count:
            flash("success", f"Added **{label}**: {chunk_count} chunks.")
        else:
            flash("warning", f"{label} has no text to index.")


def index_uploaded_files(files: list, strategy: str) -> None:
    indexed = set(list_sources())
    new_files = []
    for uploaded in files:
        if uploaded.name in indexed:
            flash("info", f"{uploaded.name} is already in the library. Remove it first to re-index it.")
        else:
            new_files.append(uploaded)
    if not new_files:
        return
    try:
        documents = load_documents(new_files)
    except ValueError as exc:
        flash("error", str(exc))
        return
    label = new_files[0].name if len(new_files) == 1 else f"{len(new_files)} documents"
    index_documents(documents, label, strategy)


def render_uploader(location: str):
    st.caption("Plain text files (.txt), up to 5 MB each.")
    return st.file_uploader(
        "Add a document",
        type=["txt"],
        accept_multiple_files=True,
        key=f"uploader_{location}_{st.session_state.uploader_version}",
        label_visibility="collapsed",
    )


def render_history_message(index: int, message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_answer_extras(message, key_prefix=f"hist_{index}")
            if show_trace:
                render_trace_panel(message)


def run_chat_turn(prompt: str) -> None:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.caption("Thinking...")
        try:
            if use_rag:
                rag_answer = ask_question(
                    to_chat_history(history), prompt, top_k=top_k, temperature=temperature
                )
                tokens = rag_answer.tokens
                chunks, search_question = rag_answer.chunks, rag_answer.search_question
            else:
                tokens = chat_stream(st.session_state.messages, temperature=temperature)
                chunks, search_question = [], None
            reply = st.write_stream(stream_once_ready(tokens, thinking.empty))
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.messages.pop()
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")
            st.session_state.messages.pop()
            return
        finally:
            thinking.empty()

        message = {
            "role": "assistant",
            "content": reply,
            "question": prompt,
            "search_question": search_question,
            "chunks": chunks,
            "top_k": top_k,
            "temperature": temperature,
        }
        render_answer_extras(message, key_prefix="latest")
        if show_trace:
            render_trace_panel(message)
    st.session_state.messages.append(message)


sources = list_sources()

with st.sidebar:
    st.header("Library")
    sidebar_files: list = []
    if sources:
        sidebar_files = render_uploader("side")
        for source in sources:
            name_column, remove_column = st.columns([5, 1], vertical_alignment="center")
            name_column.markdown(f"`{source}`")
            remove_column.button(
                "✕",
                key=f"remove_{source}",
                help=f"Remove {source} from the library",
                on_click=remove_document,
                args=(source,),
            )
        st.caption(f"{count_chunks()} chunks indexed")
    else:
        st.caption("No documents yet.")

    with st.expander("Advanced"):
        st.caption(f"Chat model: `{DEFAULT_MODEL}`")
        st.caption(f"Embedding model: `{EMBEDDING_MODEL}`")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.1)
        top_k = st.slider("Top-k chunks", 1, 8, DEFAULT_TOP_K)
        use_rag = st.toggle("Answer from my documents (RAG)", value=True)
        chunking_strategy = st.selectbox(
            "Chunking strategy",
            list(CHUNKING_STRATEGIES),
            help="Used for the next documents you add. Remove and add a document again to re-chunk it.",
        )
        show_trace = st.toggle(
            "Developer trace",
            value=False,
            help="Show what was searched and which chunks the model received.",
        )

    st.divider()
    if sources:
        with st.popover("Clear all documents", width="stretch"):
            st.warning("This removes every indexed document. You can add them again.")
            st.button(
                "Yes, clear documents",
                key="confirm_clear_documents",
                type="primary",
                on_click=clear_all_documents,
            )
    if st.session_state.messages:
        with st.popover("Clear conversation", width="stretch"):
            st.warning("This deletes the chat history. Your documents stay.")
            st.button(
                "Yes, clear conversation",
                key="confirm_clear_conversation",
                type="primary",
                on_click=clear_conversation,
            )

st.title("AI Notebook")
st.caption("Ask questions about your documents. Answers point to the chunks they came from.")

for level, text in st.session_state.flashes:
    getattr(st, level)(text)
st.session_state.flashes = []

main_files: list = []
if not sources:
    st.subheader("Add a document to begin")
    st.markdown(
        "Upload your lecture notes as `.txt` files. The linear-algebra course in "
        "`data/sample_docs/` (4 chapters, in French) is a good start."
    )
    main_files = render_uploader("main")

uploaded_files = main_files or sidebar_files
if uploaded_files:
    index_uploaded_files(uploaded_files, chunking_strategy)
    reset_library_widgets()
    st.rerun()

if st.session_state.selected_chunk:
    chat_column, preview_column = st.columns([5, 3], gap="large")
    with preview_column:
        render_preview_pane()
else:
    chat_column = st.container(key="chat_solo")

with chat_column:
    for index, message in enumerate(st.session_state.messages):
        render_history_message(index, message)

    if not st.session_state.messages and sources:
        st.caption("Try one of these:")
        with st.container(horizontal=True, horizontal_alignment="left", key="starters"):
            for index, (label, text) in enumerate(starter_prompts(sources)):
                st.button(label, key=f"starter_{index}", on_click=queue_prompt, args=(text,))

prompt = st.chat_input("Ask about your documents…") or st.session_state.pop("queued_prompt", None)

with chat_column:
    if prompt:
        if use_rag and not sources:
            st.warning("Add a document to the library first, or turn off Advanced > Answer from my documents.")
        else:
            run_chat_turn(prompt)
