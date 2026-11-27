"""STEP 3 - THE JOINT EMBEDDING SPACE: CLIP puts text and images in the SAME vector space.

Cross-modal search is then just a dot product between a text vector and an image vector.
Watch it work, then watch it fail: CLIP reads only 77 tokens of text, was trained on photos
(not charts), and cannot READ numbers in a picture. That is why step 2 exists.

Run: python src/rag/multimodal/3_clip_joint_embedding.py [--images-dir data/photos]
Needs: pip install sentence-transformers  (first run downloads the model)
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # ponytail: see 1_parse_pdf.py
from src.rag.multimodal.shared.config import ROOT, STORAGE
from src.rag.multimodal.shared.rag import clip

QUERIES = ["a bar chart", "a table of numbers", "a photograph of a wind turbine",
           "which quarter had the highest solar output", "grid uptime of Spain"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 3: CLIP joint embedding demo")
    ap.add_argument("--images-dir", type=Path, help="extra pictures (photos make the demo more interesting)")
    args = ap.parse_args()
    els = json.loads((STORAGE / "elements.json").read_text(encoding="utf-8"))["elements"]
    paths = [ROOT / e["image_path"] for e in els if e["type"] == "image"]
    if args.images_dir:
        paths += sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if not paths:
        raise SystemExit("No images: run 1_parse_pdf.py or pass --images-dir.")

    model = clip()
    iv = model.encode([Image.open(p).convert("RGB") for p in paths], normalize_embeddings=True)
    tv = model.encode(QUERIES, normalize_embeddings=True)
    print(f"image vectors {iv.shape}, text vectors {tv.shape}: same dimension = one shared space\n")
    print(" " * 46 + "".join(f"{p.name[:16]:>18}" for p in paths))
    for q, row in zip(QUERIES, tv @ iv.T):  # unit vectors: cosine == dot product
        print(f"{q[:44]:<46}" + "".join(f"{v:>18.3f}" for v in row) + f"  -> {paths[row.argmax()].name}")
    print("\nLimits: 1) text tower keeps only 77 tokens  2) scores are small and close: compare RANKS"
          "\n        3) charts are not photos: 'which quarter...' needs reading, not looking -> step 2 + fusion in step 5")


if __name__ == "__main__":
    main()
