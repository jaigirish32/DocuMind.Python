from __future__ import annotations

from functools import lru_cache
from DocuMind.bge.embedding_client import EmbeddingClient
from DocuMind.azure.chat_client import ChatClient
from DocuMind.agents.documind_agent import DocuMindAgent
from DocuMind.core.settings import get_settings
@lru_cache
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()


def get_chat_client():
    settings = get_settings()
    if settings.llm_provider == "ollama":
        from DocuMind.ollama.chat_client import OllamaChatClient
        return OllamaChatClient(model=settings.ollama_model)
    return ChatClient()


# @lru_cache
# def get_vector_store() -> WeaviateVectorStore:
#     return create_weaviate_store()

# def get_agent() -> DocuMindAgent:
#     return DocuMindAgent(
#         embedder = get_embedding_client(),
#         chat     = get_chat_client(),
#         store    = get_vector_store(),
#     )