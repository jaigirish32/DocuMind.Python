from __future__ import annotations

from functools import lru_cache
from DocuMind.azure.embedding_client import EmbeddingClient
from DocuMind.azure.chat_client import ChatClient
from DocuMind.search.azure_search_store import AzureSearchStore
from DocuMind.core.settings import get_settings


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()


@lru_cache
def get_vector_store() -> AzureSearchStore:
    return AzureSearchStore()


def get_chat_client():
    settings = get_settings()
    if settings.llm_provider == "ollama":
        from DocuMind.ollama.chat_client import OllamaChatClient
        return OllamaChatClient(model=settings.ollama_model)
    return ChatClient()