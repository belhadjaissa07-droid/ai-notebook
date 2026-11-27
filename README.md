# AI Notebook — Study Tutor (Session 3)

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

- Session 3:Multimodal RAG improves performance by using diverse data sources, enabling better understanding and more accurate responses.

  - Enhances contextual understanding by combining textual and non textual data.
  - Improves content generation with richer, more relevant and engaging outputs.
  - Increases accuracy by retrieving and using information from multiple sources

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
## Multimodal RAG Study Tutor (branch `feature/multimodal-rag`)

`src/rag/multimodal/` = numbered lesson scripts + `shared/` (the code they all use).
Search the summary, send the original.

| Step | Script | Idea | Code it leans on |
|---|---|---|---|
| 0 | `0_make_sample_pdf.py` | PDF whose answers live in charts and a table | - |
| 1 | `1_parse_pdf.py` | text / table / image / vector figure + captions | - |
| 2 | `2_caption_images.py` | vision LLM writes searchable summaries | `shared/config.py` (ask) |
| 3 | `3_clip_joint_embedding.py` | one vector space for text and images | `shared/rag.py` (clip) |
| 4 | `4_mm_ingestion_pipeline.py` | multi-vector index | `shared/rag.py` (index) |
| 5 | `5_mm_retrieval_pipeline.py` | filters + rank fusion | `shared/rag.py` (retrieve, rrf) |
| 6 | `6_mm_answer_generation.py` | text + image prompt, 3 modes | `shared/rag.py` (parts), `shared/tutor.py` |
| 7 | `7_mm_history_aware_generation.py` | tutor chat, image questions, /quiz | `shared/tutor.py` |
| 8 | `pages/1_Multimodal_RAG.py` | Streamlit Study Tutor page | `shared/tutor.py` |

```bash
git checkout -b feature/multimodal-rag
pip install -r requirements.txt -r requirements-multimodal.txt   # + GEMINI_API_KEY in .env
python src/rag/multimodal/0_make_sample_pdf.py && python src/rag/multimodal/1_parse_pdf.py
python src/rag/multimodal/2_caption_images.py && python src/rag/multimodal/4_mm_ingestion_pipeline.py
python src/rag/multimodal/6_mm_answer_generation.py -q "Which quarter had the highest solar output?" --mode text
python src/rag/multimodal/6_mm_answer_generation.py -q "Which quarter had the highest solar output?"
python src/rag/multimodal/7_mm_history_aware_generation.py     # /quiz solar, /image path question
streamlit run app.py                                            # page: Study Tutor
pytest tests/test_multimodal.py -v
```
