"""Offline checks (no API key, no model download). Run: pytest tests/test_multimodal.py -v"""
import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag.multimodal.shared import rag, tutor  # noqa: E402


def script(name):  # script names start with a digit, so `import` cannot be used
    return importlib.import_module(f"src.rag.multimodal.{name}")


def fake_embed(texts, task="RETRIEVAL_DOCUMENT", dim=256):
    """Bag-of-words hashing: similar words -> similar vectors. Stands in for Gemini."""
    out = []
    for t in texts:
        v = np.zeros(dim, dtype=np.float32)
        for w in t.lower().split():
            v[int(hashlib.md5(w.strip(".,:?()").encode()).hexdigest(), 16) % dim] += 1
        out.append((v / (np.linalg.norm(v) or 1)).tolist())
    return out


@pytest.fixture()
def elements(tmp_path, monkeypatch):
    parse = script("1_parse_pdf")
    pdf = tmp_path / "s.pdf"
    script("0_make_sample_pdf").build().save(pdf)
    monkeypatch.setattr(parse, "ROOT", tmp_path)
    return parse.parse_pdf(pdf, tmp_path / "images")


def test_parser_finds_every_modality_and_captions(elements):
    by = {e["id"]: e for e in elements}
    assert by["p3-table-1"]["caption"].startswith("Table 1") and "Morocco" in by["p3-table-1"]["content"]
    assert by["p2-image-1"]["caption"].startswith("Figure 1")      # raster chart
    assert by["p3-figure-1"]["caption"].startswith("Figure 2")     # vector chart
    assert not {"Tunisia", "Morocco", "Spain", "Italy"} & {e["content"] for e in elements if e["type"] == "text"}


def test_chunk_text_respects_size():
    chunks = rag.chunk_text("\n\n".join(["alpha " * 60, "beta " * 60, "gamma " * 200]), size=400)
    assert chunks and all(c.strip() and len(c) <= 400 for c in chunks)


def test_rrf_rewards_agreement():
    a = [{"id": "x", "score": .9}, {"id": "y", "score": .8}]
    b = [{"id": "y", "score": .3}, {"id": "z", "score": .2}]
    assert [h["id"] for h in rag.rrf([a, b])][0] == "y"


def test_index_then_retrieve_and_modality_filter(elements, tmp_path, monkeypatch):
    for e in elements:
        if e["id"] == "p2-image-1":
            e["summary"] = "Bar chart of quarterly solar and wind output in GWh. Solar peaks in Q3 at 380."
        elif e["type"] == "image":
            e["summary"] = "Bar chart of grid uptime percent by region. Spain has the highest uptime."
    monkeypatch.setattr(rag, "STORAGE", tmp_path)
    monkeypatch.setattr(rag, "embed", fake_embed)
    rag.index(elements)
    hits = rag.retrieve("which quarter had the highest solar output", 3)
    assert hits[0]["id"] == "p2-image-1" and hits[0]["doc"]["image_path"].endswith(".png")
    assert {h["type"] for h in rag.retrieve("solar output quarter", 5, ["image"])} == {"image"}


def test_parts_modes_and_image_budget(tmp_path, monkeypatch):
    from PIL import Image
    Image.new("RGB", (8, 8)).save(tmp_path / "c.png")
    monkeypatch.setattr(rag, "ROOT", tmp_path)
    img = {"id": "i", "page": 2, "doc": {"type": "image", "caption": "Figure 1", "summary": "chart text", "image_path": "c.png"}}
    pixels = lambda ps: sum(1 for p in ps if p.inline_data)  # noqa: E731
    assert pixels(rag.parts([img])[0]) == 1
    assert pixels(rag.parts([img] * 5)[0]) == 2                          # budget
    ps, _ = rag.parts([img], pixels=False)
    assert pixels(ps) == 0 and any("chart text" in (p.text or "") for p in ps)


def test_rewrite_skips_the_api_when_nothing_to_resolve():
    assert tutor.rewrite([], "Which quarter was best?") == "Which quarter was best?"
