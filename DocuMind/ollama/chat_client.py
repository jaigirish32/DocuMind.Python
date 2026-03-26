from __future__ import annotations

from openai import AsyncOpenAI

from DocuMind.core.logging.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are DocuMind, an AI document intelligence assistant.
- Financial tables appear in pipe format: "Label: value1 | value2 | value3"
- Always cite page numbers when answering questions.
- Only answer from the provided context, never from memory.
"""


class OllamaChatClient:
    """
    Ollama local LLM client.
    Uses the OpenAI-compatible API that Ollama exposes on localhost:11434.
    Drop-in replacement for ChatClient — same .ask() and .ask_raw() interface.
    """

    def __init__(self, model: str = "llama3.1:8b") -> None:
        self._client = AsyncOpenAI(
            base_url = "http://localhost:11434/v1",
            api_key  = "ollama",           # Ollama ignores this but OpenAI SDK requires it
        )
        self._deployment = model
        logger.info("OllamaChatClient ready", model=model)

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
            logger.error("Ollama chat failed", error=str(e))
            return f"Error: {str(e)}"