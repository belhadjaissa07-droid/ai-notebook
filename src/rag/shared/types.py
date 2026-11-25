"""Data shared by the retrieval and generation steps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    number: int
    distance: float
