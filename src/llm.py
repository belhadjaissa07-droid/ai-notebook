"""Session 1: Gemini Study Tutor client (ungrounded chat + streaming)."""

from __future__ import annotations

import os
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load project .env (repo root), not only cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

STUDY_TUTOR_SYSTEM_PROMPT = """You are a friendly Study Tutor for university students.
Help the learner understand concepts, work through problems step by step, and
check their reasoning. Prefer clear explanations over jargon. Ask brief
clarifying questions when the goal is ambiguous. Do not invent course materials
or citations; if you are unsure, say so and suggest how the student can verify.
Keep answers focused and educational."""

_PLACEHOLDER_KEY = "your_key_here"


def get_client() -> genai.Client:
    """Return a Gemini client, or raise if the API key is missing/placeholder."""
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key or api_key == _PLACEHOLDER_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing or still set to 'your_key_here'. "
            "Copy .env.example to .env and set a real key from Google AI Studio."
        )
    return genai.Client(api_key=api_key)


def _to_contents(
    messages: Sequence[Mapping[str, Any]],
) -> list[types.Content]:
    """Map chat roles (user/assistant) to Gemini Content (user/model)."""
    contents: list[types.Content] = []
    for message in messages:
        role = message.get("role", "user")
        text = message.get("content", "")
        if role == "assistant":
            gemini_role = "model"
        elif role == "user":
            gemini_role = "user"
        else:
            # Ignore system/tool turns; system prompt is passed via config.
            continue
        contents.append(
            types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=str(text))],
            )
        )
    return contents


def _config(temperature: float) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=STUDY_TUTOR_SYSTEM_PROMPT,
        temperature=temperature,
    )


def chat(
    messages: Sequence[Mapping[str, Any]],
    *,
    temperature: float = 0.7,
    model: str | None = None,
) -> str:
    """Blocking generate_content; returns the full assistant text."""
    client = get_client()
    response = client.models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=_to_contents(messages),
        config=_config(temperature),
    )
    return (response.text or "").strip()


def chat_stream(
    messages: Sequence[Mapping[str, Any]],
    *,
    temperature: float = 0.7,
    model: str | None = None,
) -> Generator[str, None, None]:
    """Streaming generate_content_stream; yields text chunks."""
    client = get_client()
    stream = client.models.generate_content_stream(
        model=model or DEFAULT_MODEL,
        contents=_to_contents(messages),
        config=_config(temperature),
    )
    for chunk in stream:
        text = getattr(chunk, "text", None)
        if text:
            yield text
