from __future__ import annotations
from typing import Protocol
from DocuMind.documents.models.document_chunk import DocumentChunk

class VectorStore(Protocol):
    """
    Contract for all vector database implementations.

    Any class with these methods satisfies this protocol.
    No inheritance needed.

    Current implementation : AzureSearchClient
    Future implementations  : PineconeStore, QdrantStore, FAISSStore
    """
    async def upload_documents(
        self,
        chunks:     list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upload chunks with embeddings."""
        ...

    async def hybrid_search(
        self,
        query:     str,
        embedding: list[float],
        top_k:     int,
    ) -> list[dict]:
        """Search by vector + keyword — returns list of chunk dicts."""
        ...

    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document."""
        ...

    async def list_documents(self) -> list[str]:
        """List all indexed document names."""
        ...
