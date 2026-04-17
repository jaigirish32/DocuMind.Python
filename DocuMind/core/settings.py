from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Azure OpenAI
    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_chat_deployment: str = "gpt-4o-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    # Azure Search
    azure_search_endpoint: str = ""
    azure_search_key: str = ""
    azure_search_index_name: str = "documind-pdf-generic2-index"

    # Weaviate
    weaviate_url:     str = ""
    weaviate_api_key: str = ""

    # Azure Document Intelligence
    azure_document_intelligence_endpoint: str = ""
    azure_document_intelligence_key:      str = ""

    # Gmail
    gmail_credentials_file: str = "gmail_credentials.json"
    gmail_token_file:       str = "gmail_token.json"

    llm_provider: str = "azure"   # "azure" or "ollama"
    ollama_model: str = "llama3.1:8b"

    # Upload settings
    max_upload_size_mb: int = 5        # change to 0 to disable limit

    langchain_tracing_v2: Optional[str] = None
    langchain_api_key:    Optional[str] = None
    langchain_project:    Optional[str] = None

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    secret_key: str = "changeme"
    

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()

def get_users(self) -> dict[str, str]:
    """Returns {username: password} dict"""
    result = {}
    for pair in self.users.split(","):
        pair = pair.strip()
        if ":" in pair:
            username, password = pair.split(":", 1)
            result[username.strip()] = password.strip()
    return result
