from __future__ import annotations

import json
import re
from typing import Any


def split_text(text: str, chunk_size: int = 1000, overlap: int = 120) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object")
    result = json.loads(cleaned[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def count_words(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must contain at least one non-whitespace character")
    return len(text.split())


def make_study_plan(topic: str, days: int) -> str:
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must not be empty")
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 30:
        raise ValueError("days must be an integer between 1 and 30")
    return "\n".join(
        f"Day {day}: Study {topic.strip()} and practise questions."
        for day in range(1, days + 1)
    )
