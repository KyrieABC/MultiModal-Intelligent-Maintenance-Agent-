from __future__ import annotations

import uuid

from MIMA_Agents.schemas import ManualChunk

def chunk_text(text: str, source: str, page: int, chunk_size: int=900, overlap: int = 120) -> list[ManualChunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks: list[ManualChunk]=[]
    start=0
    text=" ".join(text.split())
    while start < len(text):
        end = min(len(text), start+chunk_size)
        chunk_str= text[start:end]
        chunks.append(
            ManualChunk(
                chunk_id=str(uuid.uuid4()),
                chunk_text=chunk_str,
                source=source,
                page=page,
                metadata={"source":source,"page":page},
            )
        )
        if end==len(text):
            break
        start=end - overlap
    return chunks