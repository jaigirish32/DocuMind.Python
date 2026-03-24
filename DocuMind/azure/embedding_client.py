from __future__ import annotations

import asyncio
from DocuMind.azure.helpers import make_openai_client
from DocuMind.core.settings import get_settings
from DocuMind.core.logging.logger import get_logger
from DocuMind.core.errors.exceptions import EmbeddingError

logger = get_logger(__name__)

BATCH_SIZE = 16


class EmbeddingClient:
    """Creates embeddings using AsyncAzureOpenAI."""

    def __init__(self) -> None:
        settings         = get_settings()
        self._client     = make_openai_client(settings)
        self._deployment = settings.azure_openai_embedding_deployment

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._client.close()

    async def create_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        logger.info("Creating embeddings", total=len(texts))

        embeddings = []

        batches = [
            texts[i : i + BATCH_SIZE]
            for i in range(0, len(texts), BATCH_SIZE)
        ]

        for batch in batches:
            batch_embeddings = await self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        logger.info("Embeddings created", count=len(embeddings))
        return embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(
                input = texts,
                model = self._deployment,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error("Embedding batch failed", error=str(e))
            raise EmbeddingError(str(e))