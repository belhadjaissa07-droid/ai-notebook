"""STEP 6 - MULTIMODAL ANSWER GENERATION as the Study Tutor.

One prompt mixes modalities: [S1] label + IMAGE bytes, [S2] table, [S3] text, then the question.
Three modes prove the value of each step:
  --mode text        classic RAG: images ignored            (expect: not found)
  --mode summary     images enter as their step-2 TEXT      (works if the summary is rich)
  --mode multimodal  the ORIGINAL pixels are sent  (default)
Code: shared/rag.py (parts) and shared/tutor.py (TUTOR prompt, stream).

Run: python src/rag/multimodal/6_mm_answer_generation.py [-q "..."] [--mode text|summary|multimodal] [--clip]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # ponytail: see 1_parse_pdf.py
from src.rag.multimodal.shared.rag import parts, retrieve
from src.rag.multimodal.shared.tutor import stream

DEMO = ["Which quarter had the highest solar output?", "Which region has the highest grid uptime?",
        "How much wind capacity does Morocco have?"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 6: multimodal answer generation")
    ap.add_argument("-q", "--query")
    ap.add_argument("--mode", choices=["text", "summary", "multimodal"], default="multimodal")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--clip", action="store_true")
    args = ap.parse_args()
    for q in [args.query] if args.query else DEMO:
        hits = retrieve(q, args.k, ["text", "table"] if args.mode == "text" else None, args.clip and args.mode != "text")
        ctx, sources = parts(hits, pixels=args.mode == "multimodal")
        print(f"\nQ [{args.mode}]: {q}\ncontext: {len(sources)} sources, {sum(1 for p in ctx if p.inline_data)} image(s) as pixels\n")
        for piece in stream([], q, {"parts": ctx}):
            print(piece, end="", flush=True)
        print("\n\nSources: " + ", ".join(f"[{s['label']}] {s['type']} p{s['page']}" for s in sources))


if __name__ == "__main__":
    main()
