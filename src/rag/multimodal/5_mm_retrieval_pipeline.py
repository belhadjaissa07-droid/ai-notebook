"""STEP 5 - MULTIMODAL RETRIEVAL: search, (optionally) fuse a second retriever, resolve ids to originals.

  question -> Gemini embedding -> search mm_text ----+
           -> CLIP text tower  -> search mm_clip ----+-> Reciprocal Rank Fusion -> ids -> docstore
RRF adds 1/(60+rank) per list: items both retrievers like rise. Code: shared/rag.py (search, rrf, retrieve).

Run: python src/rag/multimodal/5_mm_retrieval_pipeline.py [-q "..."] [--only image] [--clip]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # ponytail: see 1_parse_pdf.py
from src.rag.multimodal.shared.rag import retrieve

DEMO = ["Which quarter had the highest solar output?",  # only inside Figure 1
        "Which region has the highest grid uptime?",     # only inside Figure 2
        "How much wind capacity does Morocco have?",     # only inside Table 1
        "What is planned for 2026?"]                     # plain text


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 5: multimodal retrieval")
    ap.add_argument("-q", "--query")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--only", choices=["text", "table", "image"], help="metadata filter on modality")
    ap.add_argument("--clip", action="store_true", help="add the CLIP retriever and fuse with RRF")
    args = ap.parse_args()
    for q in [args.query] if args.query else DEMO:
        print(f"\nQ: {q}")
        for rank, h in enumerate(retrieve(q, args.k, [args.only] if args.only else None, args.clip), 1):
            d = h["doc"]
            print(f"#{rank} [{h['type']:<5}] p{h['page']}  cos={h['score']:.2f} rrf={h['rrf']:.4f}  {(d['image_path'] or d['content'])[:80]!r}")


if __name__ == "__main__":
    main()
