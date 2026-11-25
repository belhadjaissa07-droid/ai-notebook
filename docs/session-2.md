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

Upload the four files of `data/sample_docs/` in the UI, then ask: *"Quelles sont les valeurs propres de Q = [[2, 1], [1, 2]] ?"*
Open **Advanced** and turn off *Answer from my documents* to see the same question answered without RAG.

## Code tour (follow the pipeline)

The numbered files in `src/rag/` have the same names, functions and order as the workshop scripts.
Shared settings and models live in `src/rag/shared/`, the Chroma helpers in `src/rag/storage/`.
The documents are loaded from the UI: `load_documents` receives the uploaded files.

| Step | Idea | File | Functions |
|---|---|---|---|
| 1 | Read the files uploaded in the UI as `Document`s | `1_ingestion_pipeline.py` | `load_documents` |
| 1 | Cut into chunks (800 chars, overlap 100) | `1_ingestion_pipeline.py` | `split_documents` |
| 1 | Embed + store in Chroma (cosine) | `1_ingestion_pipeline.py` | `create_vector_store` |
| 2 | Find the top-k closest chunks | `2_retrieval_pipeline.py` | `retrieve` |
| 3 | Put the chunks in the prompt, ask Gemini | `3_answer_generation.py` | `build_augmented_prompt`, `generate_answer` |
| 4 | Rewrite follow-ups, then retrieve + answer | `4_history_aware_generation.py` | `build_standalone_question`, `ask_question` |
| 5 | Better cutting: separators in order | `5_recursive_character_text_splitter.py` | `split_documents_recursively` |
| 7 | Let the LLM choose the cuts | `7_agentic_chunking.py` | `split_documents_agentically` |

`ask_question` in `4_history_aware_generation.py` is the whole RAG turn in three lines:
rewrite the question, retrieve, stream the answer.

## Things to try in the UI

- **Chunking strategy** (Advanced): upload the sample course with each strategy and compare chunk counts.
  Remove the documents first, because a strategy only applies to documents added afterwards.
- **Top-k**: 1 chunk may miss the answer, 8 chunks add noise.
- **Source chips**: click `[1]` to read the chunk the answer was built from.
- **Developer trace**: see the chunks the model received, in rank order.
- **Follow-up**: ask about `Q`, then *"et ses vecteurs propres ?"*. The trace shows the rewritten search question.

## Questions to try on the sample course

1. Pourquoi le produit matriciel AB est-il différent de BA ? Donne l'exemple du cours.
2. Quelle est la complexité de l'algorithme de Strassen pour le produit matriciel ?
3. Quelle est l'interprétation géométrique du déterminant d'une matrice 2x2 et 3x3 ?
4. Pourquoi la matrice M = [[2, 0, 1], [1, 3, 2], [1, 1, 1]] n'est-elle pas inversible ?
5. Quelle est la solution unique du système S par la méthode du pivot de Gauss ?
6. Que dit le théorème de Rouché-Capelli sur l'existence et l'unicité des solutions ?
7. Quelles sont les valeurs propres de la matrice Q = [[2, 1], [1, 2]] et comment calculer Q^5 ?
8. Quel est le lien entre les valeurs propres, l'ACP (PCA), PageRank et le RAG ?

## Notes

- The Gemini key is read from `GEMINI_API_KEY` (same as Session 1) and passed to LangChain explicitly.
- The agentic strategy makes one LLM call per ~4000 characters, so it is slower and uses more quota.
- Prompts are in English and ask the model to answer in the language of the question.
- `data/chroma/` is the vector database. Delete it to start fresh.
