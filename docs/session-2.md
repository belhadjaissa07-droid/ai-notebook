# Session 2 — RAG: answer from your documents

## Goal

A plain chatbot guesses about your course. RAG gives the model the right excerpts first:

```text
documents -> chunks -> embeddings -> Chroma          (ingestion, once)
question -> search Chroma -> top-k chunks -> prompt -> answer   (every question)
```

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Click **Use the sample course**, then ask: *"Quelles sont les valeurs propres de Q = [[2, 1], [1, 2]] ?"*
Open **Advanced** and turn off *Answer from my documents* to see the same question answered without RAG.

## Code tour (follow the pipeline)

The numbered files in `src/rag/` have the same names as the workshop scripts. Shared settings and models live in `src/rag/shared/`, the loader and Chroma store in `src/rag/storage/`.

| Step | Idea | File | Function |
|---|---|---|---|
| 1 | Load files as `Document`s | `storage/loader.py` | `load_text_folder`, `load_text_upload` |
| 1 | Cut into chunks (800 chars, overlap 100) | `1_ingestion_pipeline.py` | `split_by_character` |
| 1 | Embed + store in Chroma (cosine) | `storage/store.py`, `1_ingestion_pipeline.py` | `ingest_documents` |
| 2 | Find the top-k closest chunks | `2_retrieval_pipeline.py` | `retrieve` |
| 3 | Put the chunks in the prompt | `3_answer_generation.py` | `build_grounded_prompt` |
| 4 | Rewrite follow-up questions | `4_history_aware_generation.py` | `rewrite_as_standalone_question` |
| 5 | Better cutting: separators in order | `5_recursive_character_text_splitter.py` | `split_recursively` |
| 7 | Let the LLM choose the cuts | `7_agentic_chunking.py` | `split_agentically` |

`answer_question` in `4_history_aware_generation.py` is the whole RAG turn in three lines:
rewrite the question, retrieve, stream the answer.

## Things to try in the UI

- **Chunking strategy** (Advanced): add the sample course with each strategy and compare chunk counts.
  Remove the documents first, because a strategy only applies to documents added afterwards.
- **Top-k**: 1 chunk may miss the answer, 8 chunks add noise.
- **Source chips**: click `[1]` to read the chunk the answer was built from.
- **Developer trace**: see the distance of each chunk (0 = same meaning).
- **Follow-up**: ask about `Q`, then *"et ses vecteurs propres ?"*. The trace shows the rewritten search question.

## Notes

- The Gemini key is read from `GEMINI_API_KEY` (same as Session 1) and passed to LangChain explicitly.
- The agentic strategy makes one LLM call per ~4000 characters, so it is slower and uses more quota.
- Prompts are in English and ask the model to answer in the language of the question.
- `data/chroma/` is the vector database. Delete it to start fresh.
