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
        # Store the chunks as a ManualChunk object and its corresponding text embeddings as a list of float numbers
        self.records = [VectorRecord(chunk=c, embedding=v) for c, v in zip(chunks, vectors)]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        q = np.array(self.embedder.embed_query(query), dtype=float)
        scored: list[RetrievalResult] = []
        for record in self.records:
            # Calculate the cosine similarity between the query and the each embedding from each text chunk
            score = self._cosine(q, np.array(record.embedding, dtype=float))
            scored.append(RetrievalResult(chunk=record.chunk, score=score, retrieval_source="vector"))
        # rank them by similarity score
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


class PineconeVectorStore:
    def __init__(self, chunks: list[ManualChunk], embedder: OpenAIEmbedder) -> None:
        if not settings.pinecone_api_key or not Pinecone:
            raise ValueError("Pinecone is not configured.")

        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.embedder = embedder
        # create a pinecone client object
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name

        # asks Pinecone for list of indexes already in your account
        existing = {idx["name"] for idx in self.pc.list_indexes()}
        # create the index if not exist
        if self.index_name not in existing:
            # Use the dimensionality of the size of embedding of text "maintenance diagnostics"
            dim = len(self.embedder.embed_query("maintenance diagnostics"))
            # Create Pinecone index with these settings
            self.pc.create_index(
                name=self.index_name,
                dimension=dim,
                # How similarity is measured
                metric="cosine",
                # Defines how Pinecone should host the index
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
        # Perform actual semantic search:
        """
        Pinecone compares your query vector with all stored vectors
        Use cosine similarity
        return the most similar ones
        """
        response = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        results: list[RetrievalResult] = []
        # Each match is:
        """
        {
            "id": "chunk123",
            "score": 0.91,
            "metadata": {
                "text": "...",
                "source": "...",
                "page": 5
            }
        }
        """
        for match in response.get("matches", []):
            # Try to retrieve the original ManualChunk object from memory (where you stored locally: chunk = self.chunks_by_id.get(match["id"]))
            chunk = self.chunks_by_id.get(match["id"])
            # If not found (fallback): chunk wasn't stored locally or querying acrossing session
            if chunk is None:
                # Then re-build it
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
