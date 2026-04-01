from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from MIMA_Agents.Agents.state import MIMAState
from MIMA_Agents.Agents.tools import MIMATools
from MIMA_Agents.config import settings
from MIMA_Agents.Retrieval.hybrid_search import HybridRetriever


def build_mima_graph(retriever: HybridRetriever):
    
    tools = MIMATools(retriever=retriever)
    workflow = StateGraph(MIMAState)

    def vision_node(state: MIMAState) -> MIMAState:
        result = tools.vision_to_query(
            user_question=state["user_question"],
            image_path=state.get("image_path"),
        )
        return {
            "semantic_query": result["semantic_query"],
            "visual_labels": result["visual_labels"],
            "loop_count": state.get("loop_count", 0),
        }

    def retrieve_node(state: MIMAState) -> MIMAState:
        query = state.get("rewritten_query") or state["semantic_query"]
        result = tools.retrieve(query)
        diagnostics = {
            "query_used": query,
            "retrieval_confidence": result["retrieval_confidence"],
            "visual_labels": state.get("visual_labels", []),
        }
        return {
            "retrieved_contexts": result["retrieved_contexts"],
            "citations": result["citations"],
            "retrieval_confidence": result["retrieval_confidence"],
            "diagnostics": diagnostics,
        }

    def rewrite_node(state: MIMAState) -> MIMAState:
        rewritten = tools.rewrite_query(
            user_question=state["user_question"],
            prior_query=state.get("rewritten_query") or state["semantic_query"],
            retrieved_contexts=state.get("retrieved_contexts", []),
        )
        return {
            "rewritten_query": rewritten,
            "loop_count": state.get("loop_count", 0) + 1,
        }

    def web_node(state: MIMAState) -> MIMAState:
        query = state.get("rewritten_query") or state["semantic_query"]
        return {"web_context": tools.web_search(query)}

    def answer_node(state: MIMAState) -> MIMAState:
        answer = tools.generate_answer(
            user_question=state["user_question"],
            semantic_query=state.get("rewritten_query") or state["semantic_query"],
            contexts=state.get("retrieved_contexts", []),
            web_context=state.get("web_context", ""),
        )
        diagnostics = state.get("diagnostics", {})
        diagnostics["final_loop_count"] = state.get("loop_count", 0)
        return {"final_answer": answer, "diagnostics": diagnostics}

    def route_after_retrieval(state: MIMAState) -> str:
        confidence = state.get("retrieval_confidence", 0.0)
        loop_count = state.get("loop_count", 0)
        if confidence >= settings.confidence_threshold:
            return "answer"
        if loop_count < 1:
            return "rewrite"
        return "web"

    workflow.add_node("vision", vision_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("web", web_node)
    workflow.add_node("answer", answer_node)

    workflow.add_edge(START, "vision")
    workflow.add_edge("vision", "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "rewrite": "rewrite",
            "web": "web",
            "answer": "answer",
        },
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("web", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()
 