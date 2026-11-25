"""Helpers to move between plain chat messages and LangChain messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def message_text(content: Any) -> str:
    """Gemini returns either a string or a list of content blocks; keep the text."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def to_chat_history(messages: Sequence[Mapping[str, Any]]) -> list[BaseMessage]:
    """Turn the UI's {"role", "content"} dicts into LangChain messages."""
    message_types = {"user": HumanMessage, "assistant": AIMessage}
    return [message_types[message["role"]](content=message["content"]) for message in messages]
