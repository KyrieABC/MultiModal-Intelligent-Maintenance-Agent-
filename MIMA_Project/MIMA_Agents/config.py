from __future__ import annotations
"""
__future__: Postponed evaluation of type annotations
A python built-in module
•Without it:
class A:
    def foo(self) -> B:  # ERROR: B not defined yet
        pass
class B:
    pass
•With it:
from __future__ import annotations

class A:
    def foo(self) -> B:  # Works!
        pass

class B:
    pass
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

class Settings:
    # Return api key if exist, return None if not
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "mima-index")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    phoenix_collector_endpoint: str | None = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    phoenix_api_key: str | None = os.getenv("PHOENIX_API_KEY")

    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))
    top_k: int = int(os.getenv("TOP_K", "8"))

settings = Settings()