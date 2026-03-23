from typing import Protocol
from DocuMind.documents.models.document_chunk import DocumentChunk
from DocuMind.documents.models.document import Document


class ChunkBuilder(Protocol):
    def build_chunks(
        self,
        document:    Document,
        document_id: str,
    ) -> list[DocumentChunk]:
        ...