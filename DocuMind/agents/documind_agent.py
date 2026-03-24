from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

from DocuMind.bge.embedding_client import EmbeddingClient
#from DocuMind.azure.embedding_client import EmbeddingClient
from DocuMind.azure.chat_client import ChatClient
from DocuMind.core.logging.logger import get_logger
from DocuMind.search.protocols import VectorStore

logger = get_logger(__name__)


class QueryType(Enum):
    ANALYTICAL = "analytical"


@dataclass
class AskResult:
    answer:          str
    source_pages:    list[int]
    query_type:      QueryType


class DocuMindAgent:
    """
    RAG agent — simple and clean.

    3 API calls per question:
        1. Azure Embeddings — embed question
        2. Weaviate         — hybrid search
        3. GPT-4o           — generate answer
    """

    def __init__(
        self,
        embedder:         EmbeddingClient,
        chat:             ChatClient,
        store:            VectorStore,
        top_k:            int   = 20,
        mmr_top_k:        int   = 10,
        mmr_diversity:    float = 0.5,
        min_chunk_length: int   = 80,
        max_tokens:       int   = 1500,
    ) -> None:
        self._embedder         = embedder
        self._chat             = chat
        self._store            = store
        self._top_k            = top_k
        self._mmr_top_k        = mmr_top_k
        self._mmr_diversity    = mmr_diversity
        self._min_chunk_length = min_chunk_length
        self._max_tokens       = max_tokens

    async def ask(
        self,
        question:    str,
        document_id: str | None = None,
    ) -> str:
        """Simple ask — returns answer string only."""
        result = await self.ask_structured(question, document_id)
        return result.answer

    async def ask_structured(
        self,
        question:    str,
        document_id: str | None = None,
    ) -> AskResult:
        """
        Full pipeline — 3 API calls:
        embed → search → answer

        document_id — optional filter to search
        within a specific document only.
        """
        logger.info("Question", question=question[:80])

        # Step 1 — embed question
        embeddings = await self._embedder.create_embeddings([question])
        if not embeddings:
            return self._empty_result()

        # Step 2 — hybrid search — filtered by document_id if provided
        chunks = await self._store.hybrid_search(
            query       = question,
            embedding   = embeddings[0],
            top_k       = self._top_k,
            document_id = document_id,
        )
        logger.info("Retrieved", count=len(chunks))

        # Step 3 — filter short chunks
        chunks = [
            c for c in chunks
            if len(c.get("text", "")) >= self._min_chunk_length
        ]

        if not chunks:
            return self._empty_result()

        # Step 4 — MMR diversity reranking
        chunks = self._mmr(chunks, self._mmr_diversity, self._mmr_top_k)
        logger.info("After MMR", count=len(chunks))

        # Step 5 — build context and get answer
        source_pages = sorted({c["page_number"] for c in chunks})
        context = "\n\n".join(
            f"[Page {c['page_number']}]\n{c['text']}"
            for c in chunks
        )
        answer = await self._chat.ask(question, context, self._max_tokens)

        logger.info("Answer ready", pages=source_pages)

        return AskResult(
            answer       = answer,
            source_pages = source_pages,
            query_type   = QueryType.ANALYTICAL,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _mmr(
        self,
        chunks:    list[dict],
        diversity: float,
        top_k:     int,
    ) -> list[dict]:
        """MMR — prefer chunks from different pages."""
        if not chunks:
            return []

        selected   = [chunks[0]]
        seen_pages = {chunks[0]["page_number"]}
        remaining  = list(enumerate(chunks[1:], 1))

        while len(selected) < top_k and remaining:

            def score(idx_chunk: tuple) -> float:
                i, chunk        = idx_chunk
                relevance       = 1.0 / (1.0 + i)
                diversity_score = (
                    1.0 if chunk["page_number"] not in seen_pages
                    else 0.1
                )
                return (
                    (1.0 - diversity) * relevance
                    + diversity * diversity_score
                )

            best = max(remaining, key=score)
            remaining.remove(best)
            selected.append(best[1])
            seen_pages.add(best[1]["page_number"])

        return selected

    def _empty_result(self) -> AskResult:
        return AskResult(
            answer       = "I could not find relevant information. Please rephrase.",
            source_pages = [],
            query_type   = QueryType.ANALYTICAL,
        )