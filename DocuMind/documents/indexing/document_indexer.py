from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from DocuMind.core.logging.logger import get_logger
from DocuMind.core.errors.exceptions import DocumentParseError
from DocuMind.documents.readers.pdf_reader import PdfReader
from DocuMind.documents.processing.chunk_builder import ChunkBuilder
from DocuMind.bge.embedding_client import EmbeddingClient
#from DocuMind.azure.embedding_client import EmbeddingClient
from DocuMind.search.protocols import VectorStore

logger = get_logger(__name__)

class DocumentIndexer:
    """
    Orchestrates the full indexing pipeline:
        PDF → chunks → embeddings → vector store
    Uses async — non-blocking, concurrent embedding.
    """

    def __init__(
        self,
        reader:    PdfReader,
        embedder:  EmbeddingClient,
        store:     VectorStore,
    ) -> None:
        self._reader   = reader
        self._embedder = embedder
        self._store    = store

    def __init__(
        self,
        reader:    PdfReader,
        embedder:  EmbeddingClient,
        store:     VectorStore,
    ) -> None:
        self._reader   = reader
        self._embedder = embedder
        self._store    = store

    async def index(
        self,
        path:                 Path,
        document_id:          str  | None = None,
        category:             str         = "Others",
        boilerplate_patterns: list[str]   = None,
    ) -> dict:
        """
        Args:
            path                : path to PDF file
            document_id         : unique id — defaults to UUID
            boilerplate_patterns: custom filters per document type
        Returns:
            dict with indexing stats
        """
        if not path.exists():
            raise DocumentParseError(str(path), "File not found")

        # Generate document_id if not provided — use filename for predictability
        doc_id = document_id or path.stem  # path.stem = filename without extension

        logger.info(
            "Indexing document",
            path=str(path),
            document_id=doc_id,
        )

        # PdfReader is CPU bound — run in thread pool
        # so it does not block the async event loop
        loop    = asyncio.get_event_loop()
        raw_doc = await loop.run_in_executor(
            None,
            self._reader.read,
            path,
        )
        logger.info("PDF read", pages=len(raw_doc.pages))

        # ChunkBuilder is also CPU bound — run in thread pool
        builder = ChunkBuilder(
            boilerplate_patterns=boilerplate_patterns or [],
        )
        chunks = await loop.run_in_executor(
            None,
            builder.build_chunks,
            raw_doc,
            doc_id,
        )

        if not chunks:
            logger.warning("No chunks produced", path=str(path))
            return {
                "document_id": doc_id,
                "chunks":      0,
                "pages":       len(raw_doc.pages),
            }
        
        for c in chunks:
            if c.page_number == 65:
                logger.info("page number 65 chunk", t = c.text)

        logger.info("Chunks built", count=len(chunks))
        for chunk in chunks:
            chunk.category = category

        logger.info("Category set on chunks", category=category, count=len(chunks))

        # Use embedding_text for better retrieval
        # Fall back to text if embedding_text is empty
        texts = [
            chunk.embedding_text if chunk.embedding_text else chunk.text
            for chunk in chunks
        ]
        embeddings = await self._embedder.create_embeddings(texts)

        logger.info("Embeddings created", count=len(embeddings))

        await self._store.upload_documents(chunks, embeddings)

        logger.info(
            "Document indexed",
            document_id = doc_id,
            chunks      = len(chunks),
            pages       = len(raw_doc.pages),
        )

        return {
            "document_id":   doc_id,
            "document_name": Path(path.name).name,
            "chunks":        len(chunks),
            "pages":         len(raw_doc.pages),
        }