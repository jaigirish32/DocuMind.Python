from __future__ import annotations

from functools import lru_cache
from DocuMind.bge.embedding_client import EmbeddingClient
from DocuMind.azure.chat_client import ChatClient
# from DocuMind.search.factory import create_weaviate_store
# from DocuMind.search.weaviate_store import WeaviateVectorStore
from DocuMind.agents.documind_agent import DocuMindAgent

@lru_cache
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()


@lru_cache
def get_chat_client() -> ChatClient:
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