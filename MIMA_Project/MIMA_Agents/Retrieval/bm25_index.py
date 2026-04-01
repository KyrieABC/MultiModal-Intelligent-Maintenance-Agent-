from __future__ import annotations

from rank_bm25 import BM25Okapi

from MIMA_Agents.schemas import ManualChunk, RetrievalResult

class BM25Retriever:
    def __init__(self, chunks: list[ManualChunk]) -> None:
        self.chunks = chunks
        # .split() break texts into words using space
        self.corpus_tokens= [chunk.text.lower().split() for chunk in chunks]
        # BM250kapi builds a lexical retrieval model over your chunk texts
        self.index = BM25Okapi(self.corpus_tokens)
    
    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        query_tokens = query.lower().split()
        scores = self.index.get_scores(query_tokens)
        # enumerate over (index, score)
        # sort by second element(item[1])
        # Reverse=True: highest scores first
        # [:top_k]: keep only the first top_k item
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=self.chunks[idx], score=float(score), retrieval_source="bm25")
            for idx, score in ranked
        ]