"""Streamlit pieces shown under an answer: source chips, preview pane, developer trace."""

from __future__ import annotations

from typing import Any

import streamlit as st
from langchain_core.documents import Document

from src.rag import format_context
from src.ui.helpers import answer_summary, citation_chip_label


def _select_chunk(chunk: Document) -> None:
    st.session_state.selected_chunk = chunk


def _close_preview() -> None:
    st.session_state.selected_chunk = None


def render_answer_extras(message: dict[str, Any], *, key_prefix: str) -> None:
    """Numbered source chips (they open the preview pane) + a plain-language summary."""
    chunks = message.get("chunks") or []
    if chunks:
        with st.container(horizontal=True, horizontal_alignment="left", key=f"sources_{key_prefix}"):
            for index, chunk in enumerate(chunks, start=1):
                st.button(
                    citation_chip_label(index, chunk),
                    key=f"{key_prefix}_source_{index}",
                    on_click=_select_chunk,
                    args=(chunk,),
                )
    with st.expander("How I answered this"):
        st.caption(
            answer_summary(
                chunks,
                question=message["question"],
                search_question=message["search_question"],
            )
        )


def render_preview_pane() -> None:
    chunk = st.session_state.selected_chunk
    if chunk is None:
        return
    with st.container(border=True, key="preview_pane"):
        st.markdown(f"**Preview — {chunk.metadata['source']}, chunk {chunk.metadata['chunk']}**")
        st.code(chunk.page_content, language=None, wrap_lines=True)
        st.button("Close preview", key="close_preview", on_click=_close_preview)


def render_trace_panel(message: dict[str, Any]) -> None:
    """Developer trace: what was searched, what came back, what the model received."""
    with st.expander("Developer trace — retrieval", expanded=False):
        search_question = message["search_question"]
        if search_question is None:
            st.info("RAG was off — no retrieval ran.")
            return
        st.markdown(f"**Question:** {message['question']}")
        st.markdown(f"**Searched for:** {search_question}")
        st.markdown(
            f"**Settings:** top-k `{message['top_k']}` · temperature `{message['temperature']}`"
        )
        for rank, chunk in enumerate(message["chunks"], start=1):
            st.markdown(
                f"- rank **{rank}** · `{chunk.metadata['source']}` · chunk {chunk.metadata['chunk']}"
            )
        st.markdown("**Prompt context** (`format_context` output):")
        st.code(format_context(message["chunks"]), language=None, wrap_lines=True)
