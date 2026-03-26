from __future__ import annotations
from typing import Protocol
from DocuMind.documents.models.document_chunk import DocumentChunk


class VectorStore(Protocol):
    """
    Contract for all vector database implementations.
    """

    async def upload_documents(
        self,
        chunks:     list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        ...

    async def upload_email_chunks(
    self,
    chunks:     list[dict],
    embeddings: list[list[float]],
) -> None:
        ...

    async def hybrid_search(
        self,
        query:       str,
        embedding:   list[float],
        top_k:       int,
        document_id: str | None = None,
    ) -> list[dict]:
        ...

    async def search_emails(
    self,
    query:     str,
    embedding: list[float],
    top_k:     int = 10,
) -> list[dict]:
        ...

async def count_emails(self) -> int:
    ...

    async def delete_document(self, document_id: str) -> None:
        ...

    async def list_documents(self) -> list[str]:
        ...