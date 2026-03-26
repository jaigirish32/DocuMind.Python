from __future__ import annotations

from DocuMind.azure.helpers import make_openai_client
from DocuMind.core.settings import get_settings
from DocuMind.core.logging.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are DocuMind...
- Financial tables appear in pipe format: "Label: Value | Label: Value"

"""
#Example: "Total revenues: $ | 96,773" means Total revenues = $96,773 million

class ChatClient:
    """Sends questions to Azure OpenAI GPT-4o."""

    def __init__(self) -> None:
        settings         = get_settings()
        self._client     = make_openai_client(settings)
        self._deployment = settings.azure_openai_chat_deployment

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._client.close()

    async def ask(
        self,
        question:   str,
        context:    str,
        max_tokens: int = 1500,
    ) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": self._build_prompt(question, context)},
        ]
        return await self._send(messages, max_tokens)

    async def ask_raw(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return await self._send(messages, max_tokens=200)

    def _build_prompt(self, question: str, context: str) -> str:
        if context:
            return f"Context:\n{context}\n\nQuestion: {question}"
        return f"Question: {question}"

    async def _send(self, messages: list[dict], max_tokens: int) -> str:
        try:
            response = await self._client.chat.completions.create(
                model       = self._deployment,
                messages    = messages,
                max_tokens  = max_tokens,
                temperature = 0.0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Chat failed", error=str(e))
            return f"Error: {str(e)}"