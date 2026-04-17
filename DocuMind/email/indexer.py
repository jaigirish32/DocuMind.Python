from __future__ import annotations

from DocuMind.core.logging.logger import get_logger
from DocuMind.email.models import EmailMessage
#from DocuMind.bge.embedding_client import EmbeddingClient
from DocuMind.azure.embedding_client import EmbeddingClient
from DocuMind.search.protocols import VectorStore

logger = get_logger(__name__)

CHUNK_SIZE = 1000   # characters per chunk


class EmailIndexer:
    """
    Takes EmailMessage list → chunks → embeds → stores in Weaviate.
    Same concept as DocumentIndexer but for emails.

    Each email is stored as one or more chunks in Weaviate
    with email-specific metadata (sender, subject, date).
    """

    def __init__(
        self,
        embedder: EmbeddingClient,
        store:    VectorStore,
    ) -> None:
        self._embedder = embedder
        self._store    = store

    async def index(self, emails: list[EmailMessage]) -> dict:
        """
        Index a list of emails into Weaviate.
        Returns stats dict.
        """
        if not emails:
            logger.info("No emails to index")
            return {"emails": 0, "chunks": 0}

        all_chunks = []
        for email in emails:
            chunks = self._chunk_email(email)
            all_chunks.extend(chunks)

        if not all_chunks:
            return {"emails": len(emails), "chunks": 0}

        logger.info("Indexing emails", emails=len(emails), chunks=len(all_chunks))

        # Embed all chunks
        texts      = [c["text"] for c in all_chunks]
        embeddings = await self._embedder.create_embeddings(texts)

        logger.info("Email embeddings created", count=len(embeddings))

        # Store in Weaviate
        await self._store.upload_email_chunks(all_chunks, embeddings)

        logger.info("Emails indexed", emails=len(emails), chunks=len(all_chunks))

        return {
            "emails": len(emails),
            "chunks": len(all_chunks),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _chunk_email(self, email: EmailMessage) -> list[dict]:
        """
        Split one email into chunks with metadata.

        Each chunk contains:
        - text:       the actual content to embed
        - metadata:   sender, subject, date — for display in results
        - source:     "email" — so agent knows this came from email not document
        """
        if not email.body.strip():
            return []

        # Build full text — prepend subject and sender for context
        full_text = (
            f"From: {email.sender}\n"
            f"Subject: {email.subject}\n"
            f"Date: {email.date.strftime('%Y-%m-%d')}\n\n"
            f"{email.body.strip()}"
        )

        # Split into chunks
        raw_chunks = self._split(full_text)

        return [
            {
                "text":       chunk,
                "source":     "email",
                "message_id": email.message_id,
                "thread_id":  email.thread_id,
                "subject":    email.subject,
                "sender":     email.sender,
                "date":       email.date.isoformat(),
                "chunk_index": i,
            }
            for i, chunk in enumerate(raw_chunks)
        ]

    def _split(self, text: str) -> list[str]:
        """Split text into chunks of CHUNK_SIZE characters with overlap."""
        if len(text) <= CHUNK_SIZE:
            return [text]

        chunks   = []
        start    = 0
        overlap  = 100

        while start < len(text):
            end   = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - overlap

        return chunks