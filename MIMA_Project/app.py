from __future__ import annotations

import argparse
from pathlib import Path

from MIMA_Agents.Agents.graph import build_mima_graph
from MIMA_Agents.ingestion.pdf_loader import load_manual_documents
from MIMA_Agents.Observability.tracing import configure_phoenix
from MIMA_Agents.Retrieval.hybrid_search import HybridRetriever
from MIMA_Agents.schemas import AgentInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MIMA troubleshooting agent.")
    parser.add_argument("--question", required=True, help="User maintenance question.")
    parser.add_argument("--image", required=False, help="Optional path to an equipment image.")
    parser.add_argument(
        "--manual-dir",
        default="data/manuals",
        help="Directory containing technical PDF manuals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_phoenix()

    manuals = load_manual_documents(Path(args.manual_dir))
    retriever = HybridRetriever.from_documents(manuals)
    graph = build_mima_graph(retriever=retriever)

    payload = AgentInput(
        user_question=args.question,
        image_path=args.image,
    )
    result = graph.invoke(payload.model_dump())

    print("\n=== FINAL ANSWER ===\n")
    print(result.get("final_answer", "No answer generated."))

    if result.get("citations"):
        print("\n=== CITATIONS ===\n")
        for citation in result["citations"]:
            print(f"- {citation}")

    if result.get("diagnostics"):
        print("\n=== DIAGNOSTICS ===\n")
        for key, value in result["diagnostics"].items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
