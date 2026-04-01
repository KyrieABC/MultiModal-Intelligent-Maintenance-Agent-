from __future__ import annotations

from pathlib import Path

import fitz

from MIMA_Agents.ingestion.chunker import chunk_text
from MIMA_Agents.schemas import ManualChunk
from MIMA_Agents.Utilities.logging import get_logger

logger = get_logger(__name__)

def extract_text_from_pdf(pdf_path: Path) -> list[tuple[int,str]]:
    pages: list[tuple[int,str]]=[]
    with fitz.open(pdf_path) as doc:
        # enumerate give both page number, actual page content
        for index, page in enumerate(doc, start=1):
            # get_text("text"): gets the page content as plain text
            # strip(): removes extra whitespace at the beginning and end
            text = page.get_text("text").strip()
            if text:
                pages.append((index,text))
    return pages
  
# Input: folder path, Output: ManualChunk (User's custom data structure)
def load_manual_documents(manual_dir: Path) -> list[ManualChunk]:
    # Creates the directory if it doesn't exist
    # Parents=True(create parent folder if need)
    # exist_okay=True(overwrite)
    manual_dir.mkdir(parents=True,exist_ok=True)
    #glob("*.pdf"): Finds all PDF files in folder in sorted order
    pdf_files = sorted(manual_dir.glob("*.pdf"))
    all_chunks: list[ManualChunk]=[]
    for pdf in pdf_files:
        # %s: placeholder for the data you want to insert,
        # Convert insert value to string
        # In this case: %s is for pdf.name
        logger.info("Loading manual: %s", pdf.name)
        for page_num,page_text in extract_text_from_pdf(pdf):
            chunks = chunk_text(
                text=page_text,
                source=pdf.name,
                page=page_num,
            )
            all_chunks.extend(chunks)
    logger.info("Loaded %s chunks from %s PDFs", len(all_chunks),len(pdf_files))
    return all_chunks