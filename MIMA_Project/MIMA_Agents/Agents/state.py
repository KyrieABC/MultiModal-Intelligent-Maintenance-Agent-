from __future__ import annotations

from typing import Any, TypedDict

class MIMAState(TypedDict, total=False):
    user_question: str
    image_path: str | None
    semantic_query: str
    visual_labels: list[str]
    retrieved_contexts: list[dict[str, Any]]
    citations: list[str]
    retrieval_confidence: float
    rewritten_query: str
    web_context: str
    final_answer: str
    loop_count: int
    diagnostics: dict[str, Any]