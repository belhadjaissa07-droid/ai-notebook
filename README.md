# AI Notebook — Study Tutor (Session 1)

Streamlit chat UI that calls Google Gemini as an ungrounded Study Tutor.

## Features (Session 1)

- Conversation history
- System prompt (`STUDY_TUTOR_SYSTEM_PROMPT`)
- Temperature control
- Streaming replies
- Clear error if `GEMINI_API_KEY` is missing or still `your_key_here`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GEMINI_API_KEY from Google AI Studio
```

## Run

```bash
streamlit run app.py
```

## Layout

- `app.py` — Streamlit UI
- `src/llm.py` — Gemini client, `chat`, `chat_stream`
- Later sessions will add RAG, agents, multimodal (stubs only for now)
