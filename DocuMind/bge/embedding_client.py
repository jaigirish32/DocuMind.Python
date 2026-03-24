from __future__ import annotations

import asyncio
from sentence_transformers import SentenceTransformer

from DocuMind.core.logging.logger import get_logger
from DocuMind.core.errors.exceptions import EmbeddingError

logger = get_logger(__name__)

BATCH_SIZE = 16


class EmbeddingClient:
    """Creates embeddings using local BGE model."""

    def __init__(self) -> None:
        logger.info("Loading BGE model...")
        self._model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        logger.info("BGE model loaded")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def create_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        logger.info("Creating embeddings (BGE)", total=len(texts))

        embeddings = []

        batches = [
            texts[i: i + BATCH_SIZE]
            for i in range(0, len(texts), BATCH_SIZE)
        ]

        for batch in batches:
            batch_embeddings = await self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        logger.info("Embeddings created", count=len(embeddings))
        return embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True
                )
            )

            return result.tolist()

        except Exception as e:
            logger.error("Embedding batch failed", error=str(e))
            raise EmbeddingError(str(e))