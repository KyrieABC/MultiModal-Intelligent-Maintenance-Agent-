from __future__ import annotations

from typing import Iterable

from openai import OpenAI

from MIMA_Agents.config import settings

class OpenAIEmbedder:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
    
    # Output a list of text embeddings
    # (Each embedding is a list of float numbers)
    def embed_texts(self, text: Iterable[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=list(text))
        return [item.embedding for item in response.data]
    
    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
        