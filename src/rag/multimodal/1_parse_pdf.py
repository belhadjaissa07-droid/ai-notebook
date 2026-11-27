"""STEP 1 - PARSING: PDF -> typed elements (text / table / image / figure) + captions.

A text-only RAG reads a PDF as one string and drops every chart. We keep them:
  table   -> Markdown          image  -> embedded raster saved as PNG
  figure  -> VECTOR chart rendered to PNG (charts are often vector: image extraction never sees them)
  caption -> nearest 'Figure N' / 'Table N' block above or below: free context written by a human

Run: python src/rag/multimodal/1_parse_pdf.py [--pdf data/my.pdf]
Out: storage/multimodal/elements.json + images/*.png
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # ponytail: lets `python src/.../N_x.py` find `src`; drop it if you run with -m
from src.rag.multimodal.shared.config import ROOT, STORAGE

MIN_IMAGE_SIDE, MIN_FIGURE_SIDE, MARGIN, MAX_GAP = 80, 80, 24, 60
CAPTION = re.compile(r"^\s*(figure|fig\.|table)\s*\d+", re.I)


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def overlaps(a, b, ratio=0.3) -> bool:
    inter = a & b
    return not inter.is_empty and inter.get_area() / (min(a.get_area(), b.get_area()) or 1) >= ratio


def to_markdown(rows) -> str:
    rows = [[clean(str(c or "")) for c in r] for r in rows]
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines[:1] + ["|" + "---|" * len(rows[0])] + lines[1:])


def find_caption(rect, blocks) -> str:
    best, best_gap = "", MAX_GAP + 1
    for r, text in blocks:
        gaps = [g for g in (r.y0 - rect.y1, rect.y0 - r.y1) if g >= -5]  # below, above
        if CAPTION.match(text) and r.x0 < rect.x1 and r.x1 > rect.x0 and gaps and min(gaps) < best_gap:
            best, best_gap = text, min(gaps)
    return best


def parse_page(doc, page, out: Path) -> list[dict]:
    pn, count, els = page.number + 1, {}, []
    blocks = [(pymupdf.Rect(b[:4]), clean(b[4])) for b in page.get_text("blocks") if b[6] == 0 and clean(b[4])]

    def add(kind, rect, content="", path=None):
        count[kind] = count.get(kind, 0) + 1
        els.append({"id": f"p{pn}-{kind}-{count[kind]}", "type": "image" if path else kind, "page": pn,
                    "bbox": [round(v, 1) for v in rect], "content": content, "caption": "",
                    "image_path": str(path.relative_to(ROOT)) if path else None})

    for t in page.find_tables().tables:
        add("table", pymupdf.Rect(t.bbox), to_markdown(t.extract()))
    for info in page.get_image_info(xrefs=True):
        if info["xref"] and min(info["width"], info["height"]) >= MIN_IMAGE_SIDE:  # ponytail: skips images with a soft mask; handle if your PDFs have them
            pix = pymupdf.Pixmap(doc, info["xref"])
            if pix.n - pix.alpha > 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)  # CMYK -> RGB
            path = out / f"p{pn}_image_{count.get('image', 0) + 1}.png"
            pix.save(path)
            add("image", pymupdf.Rect(info["bbox"]), path=path)
    taken = [pymupdf.Rect(e["bbox"]) for e in els]
    for rect in page.cluster_drawings():  # vector charts; table borders are drawings too, hence `taken`
        if min(rect.width, rect.height) < MIN_FIGURE_SIDE or any(overlaps(rect, t) for t in taken):
            continue
        clip = (rect + (-MARGIN, -MARGIN, MARGIN, MARGIN)) & page.rect  # margin catches axis labels and legend
        path = out / f"p{pn}_figure_{count.get('figure', 0) + 1}.png"
        page.get_pixmap(clip=clip, dpi=150).save(path)
        add("figure", clip, path=path)

    areas = [pymupdf.Rect(e["bbox"]) for e in els]
    for e in els:
        e["caption"] = find_caption(pymupdf.Rect(e["bbox"]), blocks)
    for rect, text in blocks:  # text inside a table/figure belongs to it: do not index axis labels as paragraphs
        cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
        if not any(a.contains(pymupdf.Point(cx, cy)) for a in areas):
            add("text", rect, text)
    return els


def parse_pdf(pdf: Path, out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    return [e for page in doc for e in parse_page(doc, page, out)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 1: parse a PDF into typed elements")
    ap.add_argument("--pdf", type=Path, help="default: first PDF in data/")
    pdf = ap.parse_args().pdf or next(iter(sorted((ROOT / "data").glob("*.pdf"))), None)
    if not pdf or not Path(pdf).exists():
        raise SystemExit("No PDF found: put one in data/ or run 0_make_sample_pdf.py")
    els = parse_pdf(Path(pdf), STORAGE / "images")
    (STORAGE / "elements.json").write_text(json.dumps({"source": Path(pdf).name, "elements": els}, indent=1, ensure_ascii=False), encoding="utf-8")
    for e in els:
        print(f"{e['id']:<14}{e['type']:<7}p{e['page']}  {(e['image_path'] or e['content'])[:55]!r}" + (f"  [{e['caption']}]" if e["caption"] else ""))
    print(f"\n{len(els)} elements -> {STORAGE / 'elements.json'}")


if __name__ == "__main__":
    main()
