from __future__ import annotations

import weaviate
import weaviate.classes as wvc
from weaviate.client import WeaviateAsyncClient

from DocuMind.core.logging.logger import get_logger
from DocuMind.core.errors.exceptions import SearchError
from DocuMind.documents.models.document_chunk import DocumentChunk

logger = get_logger(__name__)

COLLECTION_NAME   = "DocuMindChunk"
UPLOAD_BATCH_SIZE = 100


class WeaviateVectorStore:
    """
    Weaviate implementation of VectorStore protocol.
    Uses composition — client injected from outside.
    """

    def __init__(self, client: WeaviateAsyncClient) -> None:
        self._client = client

    # ── Upload ────────────────────────────────────────────────────────────────

    async def upload_documents(
        self,
        chunks:     list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upload chunks with embeddings to Weaviate."""
        if not chunks:
            return

        logger.info("Uploading chunks", count=len(chunks))

        await self._ensure_collection()

        collection = self._client.collections.get(COLLECTION_NAME)

        objects = [
            wvc.data.DataObject(
                properties = {
                    "chunk_id":      chunk.chunk_id,
                    "document_id":   chunk.document_id,
                    "document_name": chunk.document_name,
                    "page_number":   chunk.page_number,
                    "content":       chunk.text,
                },
                vector = embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        for i in range(0, len(objects), UPLOAD_BATCH_SIZE):
            batch  = objects[i : i + UPLOAD_BATCH_SIZE]
            result = await collection.data.insert_many(batch)

            if result.has_errors:
                for err in result.errors.values():
                    logger.error("Chunk upload error", error=str(err))

        logger.info("Upload complete", chunks=len(chunks))

    # ── Search ────────────────────────────────────────────────────────────────

    async def hybrid_search(
        self,
        query:     str,
        embedding: list[float],
        top_k:     int = 10,
    ) -> list[dict]:
        """Hybrid search — BM25 + vector combined."""
        logger.info("Hybrid search", query=query[:60], top_k=top_k)

        try:
            collection = self._client.collections.get(COLLECTION_NAME)

            results = await collection.query.hybrid(
                query   = query,
                vector  = embedding,
                limit   = top_k,
                return_properties = [
                    "chunk_id",
                    "content",
                    "page_number",
                    "document_name",
                    "document_id",
                ],
                return_metadata = wvc.query.MetadataQuery(score=True),
            )

            chunks = [
                {
                    "chunk_id":      obj.properties["chunk_id"],
                    "text":          obj.properties["content"],
                    "page_number":   obj.properties["page_number"],
                    "document_name": obj.properties["document_name"],
                    "document_id":   obj.properties["document_id"],
                    "score":         obj.metadata.score,
                }
                for obj in results.objects
            ]

            logger.info("Search complete", results=len(chunks))
            return chunks

        except Exception as e:
            logger.error("Search failed", error=str(e))
            raise SearchError(str(e))

    # ── Document management ───────────────────────────────────────────────────

    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document."""
        logger.info("Deleting document", document_id=document_id)

        try:
            collection = self._client.collections.get(COLLECTION_NAME)

            await collection.data.delete_many(
                where = wvc.query.Filter.by_property(
                    "document_id"
                ).equal(document_id)
            )

            logger.info("Document deleted", document_id=document_id)

        except Exception as e:
            logger.error("Delete failed", error=str(e))
            raise SearchError(str(e))

    async def list_documents(self) -> list[str]:
        """List all unique document names."""
        try:
            collection = self._client.collections.get(COLLECTION_NAME)

            results = await collection.query.fetch_objects(
                limit             = 1000,
                return_properties = ["document_name"],
            )

            return list({
                obj.properties["document_name"]
                for obj in results.objects
            })

        except Exception as e:
            logger.error("List documents failed", error=str(e))
            return []

    # ── Private ───────────────────────────────────────────────────────────────

    async def _ensure_collection(self) -> None:
        """Create collection if it does not exist."""
        exists = await self._client.collections.exists(COLLECTION_NAME)
        if exists:
            return

        await self._client.collections.create(
            name = COLLECTION_NAME,
            properties = [
                wvc.config.Property(
                    name      = "chunk_id",
                    data_type = wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name      = "document_id",
                    data_type = wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name      = "document_name",
                    data_type = wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name      = "page_number",
                    data_type = wvc.config.DataType.INT,
                ),
                wvc.config.Property(
                    name      = "content",
                    data_type = wvc.config.DataType.TEXT,
                ),
            ],
            # We supply our own embeddings — no built-in vectorizer needed
            vector_config = wvc.config.Configure.Vectors.self_provided(),
        )

    logger.info("Collection created", name=COLLECTION_NAME)