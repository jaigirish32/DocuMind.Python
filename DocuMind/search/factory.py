from __future__ import annotations

import weaviate
from DocuMind.core.settings import get_settings
from DocuMind.search.weaviate_store import WeaviateVectorStore


def create_weaviate_store() -> WeaviateVectorStore:
    """
    Factory function — creates WeaviateVectorStore
    with injected client.

    One place for all wiring — easy to swap later.
    """
    settings = get_settings()

    client = weaviate.use_async_with_weaviate_cloud(
        cluster_url      = settings.weaviate_url,
        auth_credentials = weaviate.auth.AuthApiKey(
            settings.weaviate_api_key
        ),
    )

    return WeaviateVectorStore(client=client)