"""Steps 4-6: chunking, the multi-vector index, retrieval, rank fusion, prompt parts.

The one idea: SEARCH the summary, SEND the original.
"""
import json
import re
from functools import cache

import chromadb
from google.genai import types

from .config import ROOT, STORAGE, embed

TEXT, CLIP_COL = "mm_text", "mm_clip"


# ---- step 4: index ---------------------------------------------------------
def chunk_text(text: str, size: int = 800) -> list[str]:
    # ponytail: greedy paragraph packing, no overlap; use 5_recursive_character_text_splitter if answers straddle chunks
    chunks, cur = [], ""
    for p in filter(None, (p.strip() for p in re.split(r"\n\s*\n", text))):
        while len(p) > size:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:size])
            p = p[size:]
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n\n{p}" if cur else p
    return chunks + [cur] if cur else chunks


def build_records(elements: list[dict]) -> tuple[list[dict], dict]:
    """-> (what we EMBED, docstore of what we SEND). Tables/images are embedded as caption + summary."""
    pages, records, store = {}, [], {}
    for e in elements:
        if e["type"] == "text":
            pages.setdefault(e["page"], []).append(e["content"])
            continue
        if not e.get("summary"):
            print(f"  ! {e['id']} has no summary (run step 2): indexed by caption only")
        text = "\n".join(filter(None, [e["caption"], e.get("summary")])) or e["id"]
        records.append({"id": e["id"], "type": e["type"], "page": e["page"], "text": text})
        store[e["id"]] = e
    for page, blocks in pages.items():
        for n, c in enumerate(chunk_text("\n\n".join(blocks)), 1):
            rid = f"p{page}-chunk-{n}"
            records.append({"id": rid, "type": "text", "page": page, "text": c})
            store[rid] = {"type": "text", "page": page, "content": c}
    return records, store


def open_db():
    return chromadb.PersistentClient(path=str(STORAGE / "chroma"))


def reset(db, name):
    try:
        db.delete_collection(name)
    except Exception:  # ponytail: first run has nothing to delete; any error is treated as that
        pass
    return db.create_collection(name, metadata={"hnsw:space": "cosine"})


@cache
def clip():
    from sentence_transformers import SentenceTransformer  # heavy: only when CLIP is used
    return SentenceTransformer("clip-ViT-B-32")


def index_clip(elements: list[dict]) -> None:
    """Second index: CLIP vectors straight from image pixels. Same ids as the text index -> fusable."""
    from PIL import Image
    imgs = [e for e in elements if e["type"] == "image"]
    if not imgs:
        return
    vecs = clip().encode([Image.open(ROOT / e["image_path"]).convert("RGB") for e in imgs], normalize_embeddings=True)
    reset(open_db(), CLIP_COL).add(ids=[e["id"] for e in imgs], embeddings=vecs.tolist(),
                                   metadatas=[{"type": "image", "page": e["page"]} for e in imgs])


def index(elements: list[dict], with_clip: bool = False) -> list[dict]:
    records, store = build_records(elements)
    reset(open_db(), TEXT).add(
        ids=[r["id"] for r in records], embeddings=embed([r["text"] for r in records], "RETRIEVAL_DOCUMENT"),
        documents=[r["text"] for r in records], metadatas=[{"type": r["type"], "page": r["page"]} for r in records])
    (STORAGE / "docstore.json").write_text(json.dumps(store, indent=1, ensure_ascii=False), encoding="utf-8")
    if with_clip:
        index_clip(elements)
    return records


# ---- step 5: retrieve ------------------------------------------------------
def search(col, vec, k, where=None) -> list[dict]:
    r = col.query(query_embeddings=[vec], n_results=min(k, col.count()), where=where)
    return [{"id": i, "score": 1 - d, **m} for i, m, d in zip(r["ids"][0], r["metadatas"][0], r["distances"][0])]


def rrf(lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: fuse by RANK, because scores of different models are not comparable."""
    fused = {}
    for hits in lists:
        for rank, h in enumerate(hits, 1):
            f = fused.setdefault(h["id"], {**h, "rrf": 0.0})
            f["rrf"] += 1 / (k + rank)
    return sorted(fused.values(), key=lambda h: h["rrf"], reverse=True)


def retrieve(question: str, k: int = 4, only: list[str] | None = None, use_clip: bool = False) -> list[dict]:
    """Top-k hits, each with its ORIGINAL under hit['doc']. only=['text','table'] is classic text-only RAG."""
    db = open_db()
    store = json.loads((STORAGE / "docstore.json").read_text(encoding="utf-8"))
    where = {"type": {"$in": only}} if only else None
    lists = [search(db.get_collection(TEXT), embed([question], "RETRIEVAL_QUERY")[0], k, where)]
    if use_clip and (not only or "image" in only):
        lists.append(search(db.get_collection(CLIP_COL), clip().encode(question, normalize_embeddings=True).tolist(), k))
    return [{**h, "doc": store[h["id"]]} for h in rrf(lists)[:k]]


# ---- step 6: prompt parts --------------------------------------------------
def parts(hits: list[dict], pixels: bool = True, max_images: int = 2) -> tuple[list[types.Part], list[dict]]:
    """Interleave text and images, each source labelled [S1], [S2]...  Returns (parts, sources)."""
    out, sources, imgs = [types.Part.from_text(text="CONTEXT:")], [], 0
    for n, h in enumerate(hits, 1):
        d = h["doc"]
        cap = f" Caption: {d['caption']}" if d.get("caption") else ""
        if d["type"] == "image" and pixels and imgs < max_images:  # max_images: every image costs tokens
            imgs += 1
            out += [types.Part.from_text(text=f"[S{n}] page {h['page']}, image.{cap}"),
                    types.Part.from_bytes(data=(ROOT / d["image_path"]).read_bytes(), mime_type="image/png")]
        else:  # no pixels (text mode or budget used up): an image falls back to its text summary
            body = d.get("summary", "") if d["type"] == "image" else d["content"]
            out.append(types.Part.from_text(text=f"[S{n}] page {h['page']}, {d['type']}.{cap}\n{body}"))
        sources.append({"label": f"S{n}", "id": h["id"], "type": d["type"], "page": h["page"],
                        "image_path": d.get("image_path")})
    return out, sources
