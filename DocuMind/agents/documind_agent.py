from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from DocuMind.bge.embedding_client import EmbeddingClient
from DocuMind.azure.chat_client import ChatClient
from DocuMind.core.logging.logger import get_logger
from DocuMind.search.protocols import VectorStore

logger = get_logger(__name__)

MAX_ITERATIONS = 8

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the indexed document collection using hybrid semantic + keyword search. "
                "Always call this before answering — never answer from memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query derived from the user question.",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Optional — restrict search to one document by its ID.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve. Default 10, max 20.",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List all documents currently indexed. Use when the user asks what files are available.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

SYSTEM_PROMPT = """You are DocuMind, an AI document intelligence assistant.

Rules:
- ALWAYS use the search_documents tool before answering any question about document content.
- Never answer from memory or training data — only from retrieved chunks.
- Cite the page number for every fact you state.
- If search returns nothing relevant, say so honestly.
- Financial tables appear in pipe format: "Label: value1 | value2 | value3"
"""


class QueryType(Enum):
    ANALYTICAL = "analytical"


@dataclass
class AskResult:
    answer:       str
    source_pages: list[int]
    query_type:   QueryType
    tool_calls:   list[dict] = field(default_factory=list)


class DocuMindAgent:
    """
    AI Agent using OpenAI function calling.

    Loop:
        GPT-4o decides which tool to call
        → tool executes (search_documents / list_documents)
        → result fed back to GPT-4o
        → repeat until GPT-4o produces a final text answer
    """

    def __init__(
        self,
        embedder:         EmbeddingClient,
        chat:             ChatClient,
        store:            VectorStore,
        top_k:            int   = 20,
        mmr_top_k:        int   = 10,
        mmr_diversity:    float = 0.5,
        min_chunk_length: int   = 80,
        max_tokens:       int   = 1500,
    ) -> None:
        self._embedder         = embedder
        self._chat             = chat
        self._store            = store
        self._top_k            = top_k
        self._mmr_top_k        = mmr_top_k
        self._mmr_diversity    = mmr_diversity
        self._min_chunk_length = min_chunk_length
        self._max_tokens       = max_tokens

    async def ask(self, question: str, document_id: str | None = None) -> str:
        result = await self.ask_structured(question, document_id)
        return result.answer

    async def ask_structured(
        self,
        question:    str,
        document_id: str | None = None,
    ) -> AskResult:
        logger.info("Agent question", question=question[:80])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]

        tool_call_log: list[dict] = []
        source_pages:  list[int]  = []
        iterations = 0

        while iterations < MAX_ITERATIONS:
            iterations += 1
            logger.info("Agent iteration", n=iterations)

            response = await self._chat._client.chat.completions.create(
                model       = self._chat._deployment,
                messages    = messages,
                tools       = TOOLS,
                tool_choice = "auto",
                temperature = 0.0,
                max_tokens  = self._max_tokens,
            )

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_unset=True))

            # No tool calls → final answer
            if not msg.tool_calls:
                logger.info("Agent done", iterations=iterations, pages=source_pages)
                return AskResult(
                    answer       = msg.content or "",
                    source_pages = sorted(set(source_pages)),
                    query_type   = QueryType.ANALYTICAL,
                    tool_calls   = tool_call_log,
                )

            # Execute each tool call
            for tc in msg.tool_calls:
                fn   = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                logger.info("Tool call", tool=fn, args=args)

                tool_result = await self._execute_tool(fn, args, document_id, source_pages)
                tool_call_log.append({"tool": fn, "args": args})

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      tool_result,
                })

        logger.warning("Max iterations reached")
        return AskResult(
            answer       = "Could not produce an answer within the iteration limit.",
            source_pages = sorted(set(source_pages)),
            query_type   = QueryType.ANALYTICAL,
            tool_calls   = tool_call_log,
        )

    # ── Tool implementations ──────────────────────────────────────────────────

    async def _execute_tool(
        self,
        fn:           str,
        args:         dict,
        document_id:  str | None,
        source_pages: list[int],
    ) -> str:
        if fn == "search_documents":
            return await self._tool_search(args, document_id, source_pages)
        if fn == "list_documents":
            return await self._tool_list()
        return json.dumps({"error": f"Unknown tool: {fn}"})

    async def _tool_search(
        self,
        args:         dict,
        document_id:  str | None,
        source_pages: list[int],
    ) -> str:
        query  = args.get("query", "")
        top_k  = min(int(args.get("top_k", 10)), 20)
        doc_id = args.get("document_id") or document_id

        embeddings = await self._embedder.create_embeddings([query])
        if not embeddings:
            return json.dumps({"results": [], "message": "Embedding failed."})

        chunks = await self._store.hybrid_search(
            query       = query,
            embedding   = embeddings[0],
            top_k       = top_k,
            document_id = doc_id,
        )

        chunks = [
            c for c in chunks
            if len(c.get("text", "")) >= self._min_chunk_length
        ]

        for c in chunks:
            p = c.get("page_number")
            if p is not None:
                source_pages.append(p)

        results = [
            {
                "page":   c.get("page_number"),
                "text":   c.get("text", ""),
                "doc_id": c.get("document_id", ""),
            }
            for c in chunks
        ]

        logger.info("search_documents", returned=len(results))
        return json.dumps({"results": results})

    async def _tool_list(self) -> str:
        docs = await self._store.list_documents()
        return json.dumps({"documents": docs})