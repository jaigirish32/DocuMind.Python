from __future__ import annotations
from openai import AsyncAzureOpenAI
from DocuMind.core.settings import Settings

def make_openai_client(settings: Settings) -> AsyncAzureOpenAI:
    """
    Create AsyncAzureOpenAI client with standard settings.
    Shared by EmbeddingClient and ChatClient.
    """
    return AsyncAzureOpenAI(
        azure_endpoint = settings.azure_openai_endpoint,
        api_key        = settings.azure_openai_key,
        api_version    = "2024-08-01-preview",
        max_retries    = 5,
        timeout        = 30.0,
    )