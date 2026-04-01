from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from MIMA_Agents.config import settings
from MIMA_Agents.Retrieval.embeddings import OpenAIEmbedder
from MIMA_Agents.schemas import ManualChunk, RetrievalResult

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:  # pragma: no cover
    Pinecone = None
    ServerlessSpec = None


@dataclass
class VectorRecord:
    chunk: ManualChunk
    embedding: list[float]


class InMemoryVectorStore:
    def __init__(self, chunks: list[ManualChunk], embedder: OpenAIEmbedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        vectors = self.embedder.embed_texts([c.text for c in chunks])
        self.records = [VectorRecord(chunk=c, embedding=v) for c, v in zip(chunks, vectors)]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        q = np.array(self.embedder.embed_query(query), dtype=float)
        scored: list[RetrievalResult] = []
        for record in self.records:
            score = self._cosine(q, np.array(record.embedding, dtype=float))
            scored.append(RetrievalResult(chunk=record.chunk, score=score, retrieval_source="vector"))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


class PineconeVectorStore:
    def __init__(self, chunks: list[ManualChunk], embedder: OpenAIEmbedder) -> None:
        if not settings.pinecone_api_key or not Pinecone:
            raise ValueError("Pinecone is not configured.")

        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.embedder = embedder
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name

        existing = {idx["name"] for idx in self.pc.list_indexes()}
        if self.index_name not in existing:
            dim = len(self.embedder.embed_query("maintenance diagnostics"))
            self.pc.create_index(
                name=self.index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )

        self.index = self.pc.Index(self.index_name)
        self._upsert(chunks)

    def _upsert(self, chunks: list[ManualChunk]) -> None:
        embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(
                {
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk.text,
                        "source": chunk.source,
                        "page": chunk.page or -1,
                    },
                }
            )
        self.index.upsert(vectors=vectors)

    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        vector = self.embedder.embed_query(query)
        response = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        results: list[RetrievalResult] = []
        for match in response.get("matches", []):
            chunk = self.chunks_by_id.get(match["id"])
            if chunk is None:
                metadata = match.get("metadata", {})
                chunk = ManualChunk(
                    chunk_id=match["id"],
                    text=metadata.get("text", ""),
                    source=metadata.get("source", "unknown"),
                    page=metadata.get("page"),
                    metadata=metadata,
                )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(match.get("score", 0.0)),
                    retrieval_source="vector",
                )
            )
        return results


class VectorRetriever:
    def __init__(self, chunks: list[ManualChunk]) -> None:
        self.embedder = OpenAIEmbedder()
        if settings.pinecone_api_key and Pinecone:
            self.backend = PineconeVectorStore(chunks, self.embedder)
        else:
            self.backend = InMemoryVectorStore(chunks, self.embedder)

    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        return self.backend.search(query=query, top_k=top_k)
