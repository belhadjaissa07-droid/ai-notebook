"""Plumbing every step shares: paths, models, the Gemini client, retries, embeddings."""
import os
import time
from functools import cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env")
STORAGE = ROOT / "src" / "rag" / "storage" / "multimodal"
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768


@cache
def client() -> genai.Client:
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is missing: copy .env.example to .env and fill it in.")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def retry(fn, tries=5):
    # ponytail: retries on ANY error (429s on free keys); narrow to the SDK's error type if you need to fail fast
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  ! {type(e).__name__}, retrying in {2 ** (i + 1)}s")
            time.sleep(2 ** (i + 1))


def ask(contents, **cfg) -> str:
    """One Gemini call. `contents` is a string or a list of text / image Parts."""
    r = retry(lambda: client().models.generate_content(
        model=MODEL, contents=contents, config=types.GenerateContentConfig(**cfg)))
    return (r.text or "").strip()


def embed(texts: list[str], task: str) -> list[list[float]]:
    """task: RETRIEVAL_DOCUMENT for stored text, RETRIEVAL_QUERY for questions."""
    out = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        r = retry(lambda: client().models.embed_content(
            model=EMBED_MODEL, contents=batch,
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM)))
        out += [e.values for e in r.embeddings]
    v = np.array(out, dtype=np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).tolist()  # truncated Gemini vectors are not unit length
