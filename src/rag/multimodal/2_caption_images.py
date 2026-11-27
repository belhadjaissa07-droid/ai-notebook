"""STEP 2 - MAKE NON-TEXT SEARCHABLE: a vision LLM writes a summary of every image and table.

Our index holds TEXT vectors, so pixels must become words first. The summary is the SEARCH KEY;
the original image stays on disk and is what the answering model finally looks at (step 6).

Run: python src/rag/multimodal/2_caption_images.py [--force]     (needs GEMINI_API_KEY)
In : storage/multimodal/elements.json   Out: elements_enriched.json (+ "summary")
"""
import argparse
import json
import sys
from pathlib import Path

from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # ponytail: see 1_parse_pdf.py
from src.rag.multimodal.shared.config import MODEL, ROOT, STORAGE, ask

IMAGE_PROMPT = """You are indexing a document for a search engine. Describe this image so that someone
who cannot see it could answer questions about it.
- First line: the kind of visual (bar chart, line chart, photo, diagram...).
- Title, axis names, units, legend if visible.
- Transcribe EVERY visible number, label and category exactly as written.
- For charts: the maximum, the minimum and the main trend, one sentence each.
- Invent nothing. Plain text, at most 180 words.
Caption found in the document (may be empty): {caption}"""

TABLE_PROMPT = """You are indexing a document for a search engine. Summarise this table in at most 80 words:
topic, columns and units, and the highest / lowest values. Plain text.
Caption (may be empty): {caption}

{table}"""


def summarise(e: dict) -> str:
    if e["type"] == "image":
        img = types.Part.from_bytes(data=(ROOT / e["image_path"]).read_bytes(), mime_type="image/png")
        return ask([img, IMAGE_PROMPT.format(caption=e["caption"] or "none")], temperature=0.2)
    return ask(TABLE_PROMPT.format(caption=e["caption"] or "none", table=e["content"]), temperature=0.2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 2: summarise images and tables with Gemini")
    ap.add_argument("--force", action="store_true", help="redo summaries that already exist")
    args = ap.parse_args()
    src, out = STORAGE / "elements.json", STORAGE / "elements_enriched.json"
    if not src.exists():
        raise SystemExit("Run 1_parse_pdf.py first.")
    data = json.loads(src.read_text(encoding="utf-8"))
    old = {} if args.force or not out.exists() else {e["id"]: e.get("summary") for e in json.loads(out.read_text(encoding="utf-8"))["elements"]}
    for e in data["elements"]:
        if e["type"] != "text":
            cached = old.get(e["id"])
            e["summary"] = cached or summarise(e)
            print(f"{e['id']} [{'cached' if cached else MODEL}]\n  {e['summary']}\n")
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
