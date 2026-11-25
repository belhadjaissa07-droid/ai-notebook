# AI Notebook — Study Tutor (Session 2)

A Streamlit Study Tutor that answers from **your documents** with RAG
(Retrieval-Augmented Generation), built on Google Gemini, LangChain and Chroma.

## Features

- Session 1: chat history, system prompt, temperature, streaming
- Session 2: upload `.txt` notes (a sample linear-algebra course is in `data/sample_docs/`), chunk them, embed them,
  store them in Chroma, and get grounded answers with numbered sources
- Follow-up questions are rewritten into standalone questions before searching
- Three chunking strategies to compare: Character, Recursive, Agentic (LLM)
- Source chips open a preview of the exact chunk the answer used
- Developer trace shows the search question, the retrieved chunks and the prompt context

## Setup

Full steps for Windows, macOS and Linux, with or without a virtual environment, are in
[SETUP.md](SETUP.md). The short version:

```bash
python3 -m venv venv
source venv/bin/activate        # fish: source venv/bin/activate.fish
pip install -r requirements.txt
cp .env.example .env            # then set GEMINI_API_KEY
```

## Run

```bash
streamlit run app.py
```

Upload the four files of `data/sample_docs/` in the UI, then ask a question.
Run the offline tests (no API key needed) with `pytest`.

## Layout

```text
app.py                  Streamlit UI
src/llm/chat.py         Session 1: Gemini chat + streaming (used when RAG is off)
src/ui/theme.py         Dark theme CSS
src/ui/components.py    Source chips, preview pane, developer trace
src/ui/helpers.py       Small pure UI helpers (unit tested)
src/rag/
  1_ingestion_pipeline.py                Step 1  load_documents (uploaded files), split_documents, create_vector_store
  2_retrieval_pipeline.py                Step 2  retrieve
  3_answer_generation.py                 Step 3  build_augmented_prompt, generate_answer
  4_history_aware_generation.py          Step 4  build_standalone_question, ask_question
  5_recursive_character_text_splitter.py Step 5  split_documents_recursively
  7_agentic_chunking.py                  Step 7  split_documents_agentically
  shared/                                config.py (settings), models.py (Gemini via LangChain), messages.py
  storage/                               store.py (Chroma: open, list, count, delete)
data/sample_docs/       Sample course (4 chapters on matrices, French)
docs/session-2.md       Learner guide for this session
```

Later sessions will add multimodal RAG, agents and guardrails.
