from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

class ManualChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    page: int | None = None
    metadata: dict[str,Any]= Field(default_factory=dict)
    
class RetrievalResult(BaseModel):
    chunk: ManualChunk
    score: float
    retrieval_source: str
    
class AgentInput(BaseModel):
    user_question: str
    image_path: str | None = None
    
class AnswerWithCitations(BaseModel):
    answer:str
    citations: list[str]
    confidence: float
    

    
