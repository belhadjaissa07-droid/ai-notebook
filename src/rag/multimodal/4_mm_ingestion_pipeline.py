"""STEP 4 - MULTIMODAL INGESTION: the multi-vector pattern. SEARCH the summary, SEND the original.

   vector index (Chroma)                     docstore.json
   id -> embedding of: text chunk itself,    id -> the ORIGINAL: chunk text, table Markdown,
        table / image caption + summary           image file path
The code lives in shared/rag.py (chunk_text, build_records, index, index_clip). Read it next to this.

Run: python src/rag/multimodal/4_mm_ingestion_pipeline.py [--with-clip]   (needs steps 1-2)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # ponytail: see 1_parse_pdf.py
from src.rag.multimodal.shared.config import STORAGE
from src.rag.multimodal.shared.rag import index


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 4: build the multimodal index")
    ap.add_argument("--with-clip", action="store_true", help="also index image pixels with CLIP (step 3)")
    src = STORAGE / "elements_enriched.json"
    if not src.exists():
        raise SystemExit("Run steps 1 and 2 first.")
    records = index(json.loads(src.read_text(encoding="utf-8"))["elements"], ap.parse_args().with_clip)
    for r in records:
        print(f"{r['id']:<14}{r['type']:<7}p{r['page']}  {r['text'][:70]!r}")
    print(f"\n{len(records)} records -> {STORAGE / 'chroma'} + docstore.json")


if __name__ == "__main__":
    main()
