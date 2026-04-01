from __future__ import annotations

from typing import Any

from openai import OpenAI

from MIMA_Agents.config import settings
from MIMA_Agents.MultiModal.VisionToQuery import VisionToQuery
from MIMA_Agents.Retrieval.hybrid_search import HybridRetriever


class MIMATools:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever
        self.vision = VisionToQuery()
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def vision_to_query(self, user_question: str, image_path: str | None) -> dict[str, Any]:
        semantic_query, visual_labels = self.vision.build_semantic_query(user_question, image_path)
        return {"semantic_query": semantic_query, "visual_labels": visual_labels}

    def retrieve(self, query: str) -> dict[str, Any]:
        results, confidence = self.retriever.search(query)
        contexts = []
        citations = []
        for item in results:
            contexts.append(
                {
                    "text": item.chunk.text,
                    "source": item.chunk.source,
                    "page": item.chunk.page,
                    "score": item.score,
                }
            )
            citations.append(f"{item.chunk.source} (page {item.chunk.page})")
        return {
            "retrieved_contexts": contexts,
            "retrieval_confidence": confidence,
            "citations": sorted(set(citations)),
        }

    def rewrite_query(self, user_question: str, prior_query: str, retrieved_contexts: list[dict[str, Any]]) -> str:
        if not self.client:
            return f"{user_question}. Focus on failure symptoms, causes, replacement steps, and safety checks."

        prompt = f"""
You are a maintenance retrieval optimizer.
Rewrite the user's query to improve technical document retrieval.
Original user issue: {user_question}
Prior query: {prior_query}
Retrieved snippets: {retrieved_contexts[:3]}
Return one improved search query only.
""".strip()
        response = self.client.responses.create(
            model=settings.openai_model,
            input=prompt,
        )
        return response.output_text.strip()

    def web_search(self, query: str) -> str:
        # Placeholder adapter. Swap this with Tavily, SerpAPI, Exa, or another approved web search backend.
        return f"External web search placeholder for query: {query}"

    def generate_answer(
        self,
        user_question: str,
        semantic_query: str,
        contexts: list[dict[str, Any]],
        web_context: str = "",
    ) -> str:
        joined_context = "\n\n".join(
            f"Source: {c['source']} | Page: {c['page']}\n{c['text']}" for c in contexts[:6]
        )

        if not self.client:
            return (
                "OpenAI client not configured. Retrieved the most relevant manual sections, but answer generation "
                "requires OPENAI_API_KEY."
            )

        prompt = f"""
You are an industrial maintenance assistant.
Use only the provided evidence to answer the user's question.
If the evidence is insufficient, explicitly say so.
Provide a concise root-cause analysis, inspection checklist, and next actions.

User question: {user_question}
Semantic retrieval query: {semantic_query}

Manual evidence:
{joined_context}

Web context:
{web_context}
""".strip()
        response = self.client.responses.create(
            model=settings.openai_model,
            input=prompt,
        )
        return response.output_text.strip()
