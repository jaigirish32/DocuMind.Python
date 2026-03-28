from __future__ import annotations

import weaviate
import weaviate.classes as wvc
from weaviate.client import WeaviateAsyncClient
from weaviate.classes.query import Filter

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

    async def __aexit__(self, exc_type, exc, tb):
        # Clean shutdown
        if hasattr(self._client, "close"):
            await self._client.close()

    async def __aenter__(self):
        # Connect if not already connected
        if hasattr(self._client, "connect"):
            await self._client.connect()
        return self
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
        await self._add_category_property_if_missing()
        
        objects = [
            wvc.data.DataObject(
                properties = {
                    "chunk_id":      chunk.chunk_id,
                    "document_id":   chunk.document_id,
                    "document_name": chunk.document_name,
                    "page_number":   chunk.page_number,
                    "content":       chunk.text,
                    "category":      getattr(chunk, "category", "Others"),
                },
                vector = embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)

        ]
        logger.info("Uploading with category", category=chunks[0].category if chunks else "none")

        for i in range(0, len(objects), UPLOAD_BATCH_SIZE):
            batch  = objects[i : i + UPLOAD_BATCH_SIZE]
            result = await collection.data.insert_many(batch)

            if result.has_errors:
                for err in result.errors.values():
                    logger.error("Chunk upload error", error=str(err))

        logger.info("Upload complete", chunks=len(chunks))

    async def upload_email_chunks(
    self,
    chunks:     list[dict],
    embeddings: list[list[float]],
) -> None:
        """Upload email chunks with embeddings to Weaviate."""
        if not chunks:
            return

        logger.info("Uploading email chunks", count=len(chunks))

        await self._ensure_email_collection()

        collection = self._client.collections.get("DocuMindEmail")

        objects = [
            wvc.data.DataObject(
                properties = {
                    "message_id":  chunk["message_id"],
                    "thread_id":   chunk["thread_id"],
                    "subject":     chunk["subject"],
                    "sender":      chunk["sender"],
                    "date":        chunk["date"],
                    "content":     chunk["text"],
                    "chunk_index": chunk["chunk_index"],
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
                    logger.error("Email chunk upload error", error=str(err))

        logger.info("Email upload complete", chunks=len(chunks))

    async def search_emails(
    self,
    query:     str,
    embedding: list[float],
    top_k:     int = 10,
) -> list[dict]:
        """Hybrid search over email collection."""
        logger.info("Email search", query=query[:60], top_k=top_k)

        try:
            collection = self._client.collections.get("DocuMindEmail")

            results = await collection.query.hybrid(
                query  = query,
                vector = embedding,
                limit  = top_k,
                return_properties = [
                    "message_id",
                    "thread_id",
                    "subject",
                    "sender",
                    "date",
                    "content",
                    "chunk_index",
                ],
                return_metadata = wvc.query.MetadataQuery(score=True),
            )

            return [
                {
                    "message_id": obj.properties["message_id"],
                    "thread_id":  obj.properties["thread_id"],
                    "subject":    obj.properties["subject"],
                    "sender":     obj.properties["sender"],
                    "date":       obj.properties["date"],
                    "text":       obj.properties["content"],
                    "score":      obj.metadata.score,
                }
                for obj in results.objects
            ]

        except Exception as e:
            logger.error("Email search failed", error=str(e))
        return []


    async def count_emails(self) -> int:
        """Count total indexed email chunks."""
        try:
            collection = self._client.collections.get("DocuMindEmail")
            result     = await collection.aggregate.over_all(total_count=True)
            return result.total_count or 0
        except Exception:
            return 0

    # ── Search ────────────────────────────────────────────────────────────────

    async def hybrid_search(
    self,
    query:       str,
    embedding:   list[float],
    top_k:       int = 10,
    document_id: str | None = None,
    document_ids: list[str] | None = None,
    category:    str | None = None,
) -> list[dict]:
        """Hybrid search — BM25 + vector combined."""
        logger.info("Hybrid search", query=query[:60], top_k=top_k)

        try:
            collection = self._client.collections.get(COLLECTION_NAME)

            filters = None
            if document_id:
                filters = Filter.by_property("document_id").equal(document_id)
            elif document_ids:
                filters = Filter.by_property("document_id").contains_any(document_ids)
            elif category:
                filters = Filter.by_property("category").equal(category)

            results = await collection.query.hybrid(
                query   = query,
                vector  = embedding,
                filters = filters, # Girish
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

            if document_id:
                chunks = [c for c in chunks if c["document_id"] == document_id]

            # Trim to top_k after filtering
            chunks = chunks[:top_k]

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

    async def list_documents(self) -> list[dict]:
        """List all unique documents with name and id."""
        try:
            collection = self._client.collections.get(COLLECTION_NAME)

            results = await collection.query.fetch_objects(
                limit             = 1000,
                return_properties = ["document_name", "document_id"],
            )

            # Deduplicate by document_id
            seen = {}
            for obj in results.objects:
                doc_id   = obj.properties.get("document_id", "")
                doc_name = obj.properties.get("document_name", "")
                doc_name = doc_name.split("/")[-1].split("\\")[-1]
                category = obj.properties.get("category", "Others")
                if doc_id and doc_id not in seen:
                    seen[doc_id] = {"document_name": doc_name, "category": category}

            return [
                {
                    "document_id":   k,
                    "document_name": v["document_name"],
                    "category":      v["category"],
                }
                for k, v in seen.items()
            ]

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
                wvc.config.Property(
                    name      = "category",
                    data_type = wvc.config.DataType.TEXT,
                ),
            ],
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE,
            ),
        )

        logger.info("Collection created", name=COLLECTION_NAME)
    
    async def _add_category_property_if_missing(self) -> None:
        """Add category property to existing collection if not present."""
        try:
            collection = self._client.collections.get(COLLECTION_NAME)
            config = await collection.config.get()
            existing = [p.name for p in config.properties]
            if "category" not in existing:
                await collection.config.add_property(
                    wvc.config.Property(
                        name      = "category",
                        data_type = wvc.config.DataType.TEXT,
                    )
                )
                logger.info("Added category property to existing collection")
        except Exception as e:
            logger.warning("Could not add category property", error=str(e))

    async def _ensure_email_collection(self) -> None:
        """Create email collection if it does not exist."""
        exists = await self._client.collections.exists("DocuMindEmail")
        if exists:
            return

        await self._client.collections.create(
            name = "DocuMindEmail",
            properties = [
                wvc.config.Property(name="message_id",  data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="thread_id",   data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="subject",     data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="sender",      data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="date",        data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="content",     data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
            ],
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE,
            ),
        )
        logger.info("Email collection created")