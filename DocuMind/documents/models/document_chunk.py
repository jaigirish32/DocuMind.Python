from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DocumentChunk:
    """
    A chunk of text ready for embedding and indexing.
    Mirrors C++ DocumentChunk.h

    Two text fields — key design decision from C++ version:
      text           — clean text, stored in Azure, shown to LLM
      embedding_text — enriched with [Section:] prefix, used ONLY for embedding
      normalized_text— lowercase clean text for deduplication
    """

    chunk_id: str = ""
    document_id: str = ""
    document_name: str = ""
    page_number: int = 0
    text: str = ""
    normalized_text: str = ""
    embedding_text: str = ""
    source: str = ""
    upload_date: datetime = field(default_factory=datetime.utcnow)
