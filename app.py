"""Session 1 — Study Tutor chatbot (Streamlit + Gemini, ungrounded)."""

from __future__ import annotations

import streamlit as st
from google.genai import errors as genai_errors

from src.llm import DEFAULT_MODEL, chat_stream

st.set_page_config(page_title="AI Notebook — Study Tutor", page_icon="📓", layout="centered")
st.title("AI Notebook — Study Tutor")
st.caption("Session 1 · Ungrounded Gemini chat with history, temperature, and streaming")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    st.caption(f"Model: `{DEFAULT_MODEL}`")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Lower = more focused; higher = more creative.",
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask your Study Tutor…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(
                chat_stream(st.session_state.messages, temperature=temperature)
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": reply if isinstance(reply, str) else str(reply)}
            )
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.messages.pop()
        except (genai_errors.APIError, Exception) as exc:
            st.error(f"Gemini API error: {exc}")
            st.session_state.messages.pop()
