# Multi-Modal Intelligent Maintenance Agent (MIMA)

MIMA is an agentic multimodal RAG system for industrial equipment troubleshooting. It combines technical manuals, user-uploaded hardware photos, hybrid retrieval, reranking, and a self-correcting LangGraph workflow to produce grounded maintenance guidance.

## What this project demonstrates

- **Multimodal troubleshooting** with equipment manuals + hardware images
- **Vision-to-Query** using a CLIP model to convert images into semantic search cues
- **Hybrid retrieval** using BM25 + vector search with Pinecone support
- **Cross-encoder reranking** for precision improvement
- **Self-correcting ReAct-style loop** that reformulates low-confidence queries or triggers web search
- **Observability** hooks for Arize Phoenix tracing
- **Evaluation** hooks for RAGAS metrics such as faithfulness and context precision

## Project structure

```text
mima_project/
├── README.md
├── requirements.txt
├── .env.example
├── app.py
├── data/
│   └── manuals/
└── mima/
    ├── __init__.py
    ├── config.py
    ├── schemas.py
    ├── agents/
    │   ├── __init__.py
    │   ├── graph.py
    │   ├── state.py
    │   └── tools.py
    ├── evaluation/
    │   ├── __init__.py
    │   └── ragas_eval.py
    ├── ingestion/
    │   ├── __init__.py
    │   ├── chunker.py
    │   └── pdf_loader.py
    ├── multimodal/
    │   ├── __init__.py
    │   └── vision_to_query.py
    ├── observability/
    │   ├── __init__.py
    │   └── phoenix_tracing.py
    ├── retrieval/
    │   ├── __init__.py
    │   ├── bm25_index.py
    │   ├── embeddings.py
    │   ├── hybrid_search.py
    │   ├── reranker.py
    │   └── vector_store.py
    └── utils/
        ├── __init__.py
        └── logging.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in the required environment variables in `.env`.

## Environment variables

- `OPENAI_API_KEY`
- `PINECONE_API_KEY` (optional if using local in-memory vector fallback)
- `PINECONE_INDEX_NAME` (optional)
- `PHOENIX_COLLECTOR_ENDPOINT` (optional)
- `PHOENIX_API_KEY` (optional)
- `MIMA_WEB_SEARCH_API_KEY` (optional placeholder if you wire a real web search backend)

## Running the demo

1. Place PDF manuals into `data/manuals/`
2. Optionally add an image path to test multimodal input
3. Run:

```bash
python app.py \
  --question "The hydraulic arm is overheating and making a grinding noise. What should I inspect?" \
  --image path/to/equipment_photo.jpg
```

## Example flow

1. Load manuals and chunk them
2. Build BM25 and vector indexes
3. Convert image to semantic fault labels using CLIP
4. Merge user question + visual labels into a retrieval query
5. Retrieve with BM25 + vector search
6. Rerank with a cross-encoder
7. Score retrieval confidence
8. If confidence < 0.8, reformulate or web-search
9. Produce final answer with grounded citations
10. Optionally evaluate outputs with RAGAS and inspect traces in Phoenix

## Notes

- This implementation is built to be **portfolio-ready** and modular.
- Some production integrations are implemented as **clean adapters** with graceful fallbacks.
- I applied fork currently



