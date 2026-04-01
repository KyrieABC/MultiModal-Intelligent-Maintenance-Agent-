from __future__ import annotations

from collections import defaultdict

from MIMA_Agents.config import settings
from MIMA_Agents.Retrieval.bm25_index import BM25Retriever
from MIMA_Agents.Retrieval.reranker import CrossEncoderReranker
from MIMA_Agents.Retrieval.vector_store import VectorRetriever
from MIMA_Agents.schemas import ManualChunk, RetrievalResult


class HybridRetriever:
    def __init__(self, chunks: list[ManualChunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.vector = VectorRetriever(chunks)
        self.reranker = CrossEncoderReranker()

    @classmethod
    def from_documents(cls, chunks: list[ManualChunk]) -> "HybridRetriever":
        return cls(chunks)

    def _fuse(self, bm25_results: list[RetrievalResult], vector_results: list[RetrievalResult]) -> list[RetrievalResult]:
        merged: dict[str, RetrievalResult] = {}
        scores: dict[str, float] = defaultdict(float)

        for rank, item in enumerate(bm25_results, start=1):
            merged[item.chunk.chunk_id] = item
            scores[item.chunk.chunk_id] += 1.0 / (50 + rank)

        for rank, item in enumerate(vector_results, start=1):
            merged[item.chunk.chunk_id] = item
            scores[item.chunk.chunk_id] += 1.0 / (50 + rank)

        fused = [
            RetrievalResult(
                chunk=merged[chunk_id].chunk,
                score=score,
                retrieval_source="hybrid_fused",
            )
            for chunk_id, score in scores.items()
        ]
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

    def retrieval_confidence(self, results: list[RetrievalResult]) -> float:
        if not results:
            return 0.0
        top_scores = [item.score for item in results[:3]]
        confidence = sum(top_scores) / max(len(top_scores), 1)
        return max(0.0, min(1.0, confidence))

    def search(self, query: str, top_k: int | None = None) -> tuple[list[RetrievalResult], float]:
        k = top_k or settings.top_k
        bm25_results = self.bm25.search(query, top_k=k)
        vector_results = self.vector.search(query, top_k=k)
        fused = self._fuse(bm25_results, vector_results)
        reranked = self.reranker.rerank(query, fused, top_k=k)
        confidence = self.retrieval_confidence(reranked)
        return reranked, confidence
