"""STEP 8 - the Study Tutor page. Streamlit turns every file in pages/ into a sidebar page: app.py is untouched.
All logic is prepare() + stream() from shared/tutor.py; this file only draws."""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # ponytail: see 1_parse_pdf.py
from src.rag.multimodal.shared.config import STORAGE
from src.rag.multimodal.shared.tutor import prepare, stream

st.set_page_config(page_title="Study Tutor", page_icon="🎓", layout="wide")
st.title("🎓 Study Tutor")
st.caption("Ask about the document. The tutor reads text, tables AND charts, and shows you where.")
if not (STORAGE / "docstore.json").exists():
    st.error("No index yet: run steps 1, 2 and 4 in src/rag/multimodal/ first.")
    st.stop()

with st.sidebar:
    k = st.slider("Sources (k)", 2, 8, 4)
    use_clip = st.checkbox("Add CLIP retriever (index with --with-clip)")
    upload = st.file_uploader("Attach an image to your next question", type=["png", "jpg", "jpeg", "webp"])
    quiz = st.button("Quiz me")
    if st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()
chat = st.session_state.setdefault("chat", [])


def show_sources(sources, query):
    with st.expander(f"Sources ({len(sources)}) - search: {query}"):
        for s in sources:
            st.markdown(f"**[{s['label']}]** {s['type']} p{s['page']} `{s['id']}`")
            if s["image_path"]:
                st.image(str(ROOT / s["image_path"]), width=420)


for m in chat:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("sources"):
            show_sources(m["sources"], m["query"])

question = st.chat_input("Ask the tutor") or ("Quiz me on the key facts in this document. Ask one question." if quiz else None)
if question:
    image = (upload.getvalue(), upload.type) if upload else None
    history = [{"role": m["role"], "text": m["text"]} for m in chat]
    with st.chat_message("user"):
        st.markdown(question)
        if upload:
            st.image(upload, width=240)
    with st.chat_message("assistant"):
        with st.spinner("Looking through text, tables and figures..."):
            turn = prepare(history, question, image, k, use_clip)
        answer = st.write_stream(stream(history, question, turn, image))
        show_sources(turn["sources"], turn["query"])
    chat += [{"role": "user", "text": question},
             {"role": "assistant", "text": answer, "sources": turn["sources"], "query": turn["query"]}]
