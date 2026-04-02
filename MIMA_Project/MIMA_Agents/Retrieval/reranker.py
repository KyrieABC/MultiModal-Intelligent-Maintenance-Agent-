from __future__ import annotations

from sentence_transformers import CrossEncoder

from MIMA_Agents.schemas import RetrievalResult


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int = 8) -> list[RetrievalResult]:
        pairs = [(query, item.chunk.text) for item in candidates]
        if not pairs:
            return []
        scores = self.model.predict(pairs)
        reranked: list[RetrievalResult] = []
        # zip pair each candidate(element) with its new score
        for item, score in zip(candidates, scores):
            reranked.append(
                RetrievalResult(
                    chunk=item.chunk,
                    score=float(score),
                    retrieval_source=f"{item.retrieval_source}+rerank",
                )
            )
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
