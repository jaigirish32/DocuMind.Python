from __future__ import annotations

from openai import AsyncAzureOpenAI
from DocuMind.core.settings import get_settings
from DocuMind.core.logging.logger import get_logger
from DocuMind.core.errors.exceptions import EmbeddingError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from langsmith import traceable

logger = get_logger(__name__)

BATCH_SIZE = 16
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSIONS = 1536


class EmbeddingClient:
    """Creates embeddings using Azure OpenAI text-embedding-3-small."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key,
            api_version="2024-02-01",
        )
        self._deployment = settings.azure_openai_embedding_deployment
        logger.info("Azure embedding client initialized", model=EMBEDDING_MODEL)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    @traceable(name="AzureEmbedding.create")
    async def create_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        logger.info("Creating embeddings (Azure)", total=len(texts))

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

    @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(
                input=texts,
                model=self._deployment,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.warning("Embedding batch failed, will retry", error=str(e))
            raise EmbeddingError(str(e))