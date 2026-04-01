from __future__ import annotations

import os
from MIMA_Agents.config import settings
from MIMA_Agents.Utilities.logging import get_logger

logger = get_logger(__name__)
"""
What the __name__ would be:
if another file: 
from MIMA_Agents.Observability.tracing import ...
Then:   __name__ == "MIMA_Agents.Observability.tracing"
if run directly:
python tracing.py
Then: __name__== "__main__"
"""
"""
In large systems (pipelines):
•This automatically labels logs by module
Ex: 
2026-03-31 | INFO | MIMA_Agents.Observability.tracing | Starting trace pipeline
"""
#print(__name__)

def configure() -> None:
    if not settings.phoenix_collector_endpoint:
        logger.info("Phoenix tracing not configured; continuing without tracing.")
        return

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = settings.phoenix_collector_endpoint
    if settings.phoenix_api_key:
        os.environ["PHOENIX_API_KEY"] = settings.phoenix_api_key

    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument()
        logger.info("Phoenix/OpenInference tracing enabled.")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to enable Phoenix tracing: %s", exc)
