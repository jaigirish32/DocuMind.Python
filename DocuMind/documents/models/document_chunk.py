from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class DocumentChunk:
    chunk_id:       str = ""
    document_id:    str = ""
    document_name:  str = ""
    page_number:    int = 0
    text:           str = ""
    normalized_text:str = ""
    embedding_text: str = ""
    source:         str = ""
    category:       str = "Others"
    user_id:         str = ""
    upload_date:    datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))