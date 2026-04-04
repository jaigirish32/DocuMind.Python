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
                "Search uploaded PDF documents and files using hybrid semantic + keyword search. "
                "Use this for questions about contracts, reports, manuals, financial statements, "
                "or any uploaded file content. "
                "Never use this for questions about emails, messages, or conversations."
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
            "name": "search_emails",
            "description": (
                "Search indexed emails from Gmail or Outlook using hybrid semantic + keyword search. "
                "Use this for questions about emails, messages, conversations, or anything "
                "a person said, sent, or communicated. "
                "Also use this when the question mentions a person's name, "
                "sender, recipient, or any communication context even without the word 'email'. "
                "Examples: 'What did John say about payment?', 'Did anyone mention the deadline?', "
                "'What was discussed about the invoice?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query derived from the user question.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of email chunks to retrieve. Default 10, max 20.",
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
            "description": "List all uploaded documents currently indexed. Use when the user asks what files are available.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

SYSTEM_PROMPT = """You are DocuMind, an AI document and email intelligence assistant.

You have access to two knowledge sources:
1. Uploaded documents (PDFs, reports, contracts) — search with search_documents
2. Indexed emails (Gmail, Outlook) — search with search_emails

Rules:
- ALWAYS use the appropriate tool before answering — never answer from memory.
- For document questions → use search_documents
- For email/communication questions → use search_emails
- For questions that could involve both → use BOTH tools and combine the results
- If someone asks about what a person said or communicated → use search_emails
- Cite page numbers for documents and sender/date for emails.
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
    Searches both documents and emails based on question context.
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
        document_ids: list[str] | None = None,
        history:     list[dict] = [],
    ) -> AskResult:
        logger.info("Agent question", question=question[:80])

        system = SYSTEM_PROMPT
        if document_id:
            system += f"\n\nIMPORTANT: The user has selected document '{document_id}'. ALWAYS search only this document unless the user explicitly asks about another document. Pass document_id='{document_id}' to every search_documents call."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user",   "content": question},
        ]

        tool_call_log: list[dict] = []
        source_pages:  list[int]  = []
        iterations = 0

        while iterations < MAX_ITERATIONS:
            iterations += 1
            logger.info("Agent iteration", n=iterations)

            # When document is selected — remove list_documents tool
            active_tools = (
                [t for t in TOOLS if t["function"]["name"] != "list_documents"]
                if document_id else TOOLS
            )

            tool_choice = (
                {"type": "function", "function": {"name": "search_documents"}}
                if document_id and iterations == 1
                else "auto"
            )

            response = await self._chat._client.chat.completions.create(
                model       = self._chat._deployment,
                messages    = messages,
                tools       = TOOLS,
                tool_choice = tool_choice,
                temperature = 0.0,
                max_tokens  = self._max_tokens,
            )

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_unset=True))

            if not msg.tool_calls:
                logger.info("Agent done", iterations=iterations, pages=source_pages)
                return AskResult(
                    answer       = msg.content or "",
                    source_pages = sorted(set(source_pages)),
                    query_type   = QueryType.ANALYTICAL,
                    tool_calls   = tool_call_log,
                )

            for tc in msg.tool_calls:
                fn   = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                logger.info("Tool call", tool=fn, args=args)

                tool_result = await self._execute_tool(fn, args, document_id,document_ids, source_pages)
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
    document_ids: list[str] | None,
    source_pages: list[int],
) -> str:
        if fn == "search_documents":
            return await self._tool_search_documents(args, document_id, document_ids,source_pages)
        if fn == "search_emails":
            return await self._tool_search_emails(args)
        if fn == "list_documents":
            if document_id:
                # When document is selected — don't list all docs, search selected one
                args["query"] = "overview contents summary"
                return await self._tool_search_documents(args, document_id, document_ids,source_pages)
            return await self._tool_list()
        return json.dumps({"error": f"Unknown tool: {fn}"})

    async def _tool_search_documents(
        self,
        args:         dict,
        document_id:  str | None,
        document_ids: list[str] | None,
        source_pages: list[int],
    ) -> str:
        query  = args.get("query", "")
        top_k  = min(int(args.get("top_k", 10)), 20)
        doc_id = args.get("document_id") or document_id

        embeddings = await self._embedder.create_embeddings([query])
        if not embeddings:
            return json.dumps({"results": [], "message": "Embedding failed."})

        chunks = await self._store.hybrid_search(
            query        = query,
            embedding    = embeddings[0],
            top_k        = top_k,
            document_id  = doc_id,
            document_ids = document_ids if not doc_id else None,
        )

        chunks = [
            c for c in chunks
            if len(c.get("content", "")) >= self._min_chunk_length
        ]

        for c in chunks:
            p = c.get("page_number")
            if p is not None:
                source_pages.append(p)

        results = [
            {
                "page":   c.get("page_number"),
                "text":   c.get("content", ""),
                "doc_id": c.get("document_id", ""),
            }
            for c in chunks
        ]

        logger.info("search_documents", returned=len(results))
        return json.dumps({"results": results})

    async def _tool_search_emails(self, args: dict) -> str:
        query = args.get("query", "")
        top_k = min(int(args.get("top_k", 10)), 20)

        embeddings = await self._embedder.create_embeddings([query])
        if not embeddings:
            return json.dumps({"results": [], "message": "Embedding failed."})

        results = await self._store.search_emails(
            query     = query,
            embedding = embeddings[0],
            top_k     = top_k,
        )

        logger.info("search_emails", returned=len(results))
        return json.dumps({"results": results})

    async def _tool_list(self) -> str:
        docs = await self._store.list_documents()
        return json.dumps({"documents": docs})