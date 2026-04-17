from __future__ import annotations

import uuid
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

from DocuMind.core.settings import get_settings
from DocuMind.core.logging.logger import get_logger
from DocuMind.documents.models.document_chunk import DocumentChunk
from langsmith import traceable

logger = get_logger(__name__)

DIMENSIONS = 1536


class AzureSearchStore:
    """Vector store using Azure AI Search."""

    def __init__(self) -> None:
        settings = get_settings()
        self._endpoint   = settings.azure_search_endpoint
        self._credential = AzureKeyCredential(settings.azure_search_key)
        self._index_name = settings.azure_search_index_name

    def _get_search_client(self) -> SearchClient:
        return SearchClient(
            endpoint    = self._endpoint,
            index_name  = self._index_name,
            credential  = self._credential,
        )

    def _get_index_client(self) -> SearchIndexClient:
        return SearchIndexClient(
            endpoint   = self._endpoint,
            credential = self._credential,
        )

    async def ensure_index(self) -> None:
        """Create index if it doesn't exist."""
        async with self._get_index_client() as client:
            try:
                await client.get_index(self._index_name)
                logger.info("Azure Search index exists", index=self._index_name)
            except Exception:
                await self._create_index(client)

    async def _create_index(self, client: SearchIndexClient) -> None:
        fields = [
            SimpleField(
                name="chunkId",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SimpleField(
                name="documentId",
                type=SearchFieldDataType.String,
                filterable=True,
                retrievable=True,
            ),
            SearchableField(
                name="documentName",
                type=SearchFieldDataType.String,
                filterable=True,
                retrievable=True,
            ),
            SimpleField(
                name="category",
                type=SearchFieldDataType.String,
                filterable=True,
                retrievable=True,
            ),
            SimpleField(
                name="pageNumber",
                type=SearchFieldDataType.Int32,
                filterable=True,
                retrievable=True,
            ),
            # ── NEW — userId field ────────────────────────────────
            SimpleField(
                name="userId",
                type=SearchFieldDataType.String,
                filterable=True,     # needed for WHERE userId = 'X'
                retrievable=True,
            ),
            # ─────────────────────────────────────────────────────
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                retrievable=True,
                analyzer_name="en.microsoft",
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                retrievable=True,
                vector_search_dimensions=DIMENSIONS,
                vector_search_profile_name="vector-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config",
            )],
        )

        index = SearchIndex(
            name         = self._index_name,
            fields       = fields,
            vector_search = vector_search,
        )

        await client.create_index(index)
        logger.info("Azure Search index created", index=self._index_name)

    async def upload_documents(
        self,
        chunks:     list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            documents.append({
                "chunkId":      str(uuid.uuid4()),
                "documentId":   chunk.document_id,
                "documentName": chunk.document_name,
                "category":     getattr(chunk, "category", "Others"),
                "pageNumber":   chunk.page_number,
                "userId":       getattr(chunk, "user_id", ""),  # ← ADDED
                "content":      chunk.text,
                "embedding":    embedding,
            })

        async with self._get_search_client() as client:
            await client.upload_documents(documents=documents)
            logger.info(
                "Documents uploaded to Azure Search",
                count=len(documents),
            )

    async def upload_email_chunks(
        self,
        chunks:     list[dict],
        embeddings: list[list[float]],
    ) -> None:
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            documents.append({
                "chunkId":      str(uuid.uuid4()),
                "documentId":   chunk.get("email_id", ""),
                "documentName": chunk.get("subject", ""),
                "category":     "Emails",
                "pageNumber":   0,
                "userId":       chunk.get("user_id", ""),  # ← ADDED
                "content":      chunk.get("content", ""),
                "embedding":    embedding,
            })

        async with self._get_search_client() as client:
            await client.upload_documents(documents=documents)
            logger.info(
                "Email chunks uploaded to Azure Search",
                count=len(documents),
            )

    @traceable(name="AzureAISearch.hybrid")
    async def hybrid_search(
        self,
        query:        str,
        embedding:    list[float],
        top_k:        int,
        document_id:  str | None = None,
        document_ids: list[str] | None = None,
        user_id:      str | None = None,   # ← ADDED
    ) -> list[dict]:
        vector_query = VectorizedQuery(
            vector              = embedding,
            k_nearest_neighbors = top_k,
            fields              = "embedding",
        )

        # Build filter — userId ALWAYS included if provided
        # This ensures users only see their own documents
        filters = []

        if user_id:
            filters.append(f"userId eq '{user_id}'")

        if document_ids and len(document_ids) > 0:
            ids = " or ".join(
                [f"documentId eq '{did}'" for did in document_ids]
            )
            filters.append(f"({ids})")
        elif document_id:
            filters.append(f"documentId eq '{document_id}'")

        filter_expr = " and ".join(filters) if filters else None

        async with self._get_search_client() as client:
            results = await client.search(
                search_text    = query,
                vector_queries = [vector_query],
                filter         = filter_expr,
                top            = top_k,
                select         = [
                    "documentId", "documentName",
                    "category", "pageNumber", "content",
                ],
            )

            chunks = []
            async for result in results:
                chunks.append({
                    "document_id":   result["documentId"],
                    "document_name": result["documentName"],
                    "category":      result.get("category", "Others"),
                    "page_number":   result.get("pageNumber", 0),
                    "content":       result["content"],
                    "score":         result.get("@search.score", 0),
                    "embedding":     [],
                })

            return chunks

    async def search_emails(
        self,
        query:     str,
        embedding: list[float],
        top_k:     int = 10,
        user_id:   str | None = None,   # ← ADDED
    ) -> list[dict]:
        vector_query = VectorizedQuery(
            vector              = embedding,
            k_nearest_neighbors = top_k,
            fields              = "embedding",
        )

        # Filter by Emails AND userId
        filters = ["category eq 'Emails'"]
        if user_id:
            filters.append(f"userId eq '{user_id}'")
        filter_expr = " and ".join(filters)

        async with self._get_search_client() as client:
            results = await client.search(
                search_text    = query,
                vector_queries = [vector_query],
                filter         = filter_expr,
                top            = top_k,
                select         = [
                    "documentId", "documentName",
                    "content", "pageNumber",
                ],
            )

            chunks = []
            async for result in results:
                chunks.append({
                    "email_id": result["documentId"],
                    "subject":  result["documentName"],
                    "content":  result["content"],
                    "score":    result.get("@search.score", 0),
                })

            return chunks

    async def count_emails(self, user_id: str | None = None) -> int:
        filters = ["category eq 'Emails'"]
        if user_id:
            filters.append(f"userId eq '{user_id}'")

        async with self._get_search_client() as client:
            results = await client.search(
                search_text        = "*",
                filter             = " and ".join(filters),
                include_total_count = True,
                top                = 0,
            )
            count = 0
            async for _ in results:
                count += 1
            return count

    async def delete_document(
        self,
        document_id: str,
        user_id:     str | None = None,   # ← ADDED safety check
    ) -> None:
        # Only delete if document belongs to this user
        filters = [f"documentId eq '{document_id}'"]
        if user_id:
            filters.append(f"userId eq '{user_id}'")

        async with self._get_search_client() as client:
            results = await client.search(
                search_text = "*",
                filter      = " and ".join(filters),
                select      = ["chunkId"],
                top         = 1000,
            )

            chunk_ids = []
            async for result in results:
                chunk_ids.append({"chunkId": result["chunkId"]})

            if chunk_ids:
                await client.delete_documents(documents=chunk_ids)
                logger.info(
                    "Document deleted from Azure Search",
                    document_id = document_id,
                    chunks      = len(chunk_ids),
                )

    async def list_documents(
        self,
        user_id: str | None = None,   # ← ADDED
    ) -> list[dict]:
        # Filter by user — only return their documents
        filters = ["category ne 'Emails'"]
        if user_id:
            filters.append(f"userId eq '{user_id}'")

        async with self._get_search_client() as client:
            results = await client.search(
                search_text = "*",
                select      = ["documentId", "documentName", "category"],
                filter      = " and ".join(filters),
                top         = 1000,
            )

            seen = {}
            async for result in results:
                doc_id = result["documentId"]
                if doc_id and doc_id not in seen:
                    seen[doc_id] = {
                        "document_id":   doc_id,
                        "document_name": result["documentName"],
                        "category":      result.get("category", "Others"),
                    }

            return list(seen.values())