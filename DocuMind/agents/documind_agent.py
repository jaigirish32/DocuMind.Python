from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

from DocuMind.azure.chat_client import ChatClient
from DocuMind.core.logging.logger import get_logger
from DocuMind.search.protocols import VectorStore
from DocuMind.core.token_counter import validate_question, truncate_chunks_to_limit
from DocuMind.core.reranker import mmr_rerank
from langsmith import traceable

logger = get_logger(__name__)

MAX_ITERATIONS = 5

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
- If search returns nothing relevant, say so honestly.
- Financial tables appear in pipe format: "Label: value1 | value2 | value3"

Citation rules:
- Each search result contains a "page" field — use it to cite sources.
- At the end of your answer always write: Sources: page X
- ONLY cite the page numbers that DIRECTLY contain the information you used.
- If answer came from page 4 only → write Sources: page 4
- If answer came from pages 1 and 4 → write Sources: pages 1, 4
- Do NOT cite pages that were retrieved but not used in your answer.
- Never cite more than 3 pages unless the answer truly spans all of them.
- NEVER approximate or estimate values not explicitly
  stated in the retrieved chunks.
- If a specific value is not found in the search results,
  say exactly: "This information is not available in
  the retrieved documents."
- Do not use phrases like "approximately", "based on context",
  or "estimated" — these indicate hallucination.
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
        embedder:         object,
        chat:             ChatClient,
        store:            VectorStore,
        top_k:            int   = 10,
        mmr_top_k:        int   = 5,
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

    # ── Public ────────────────────────────────────────────────────────────────

    async def ask(self, question: str, document_id: str | None = None) -> str:
        result = await self.ask_structured(question, document_id)
        return result.answer

    @traceable(name="DocuMind.ask")
    async def ask_structured(
        self,
        question:     str,
        document_id:  str | None = None,
        document_ids: list[str] | None = None,
        history:      list[dict] = [],
        user_id:      str | None = None,        # ← ADDED
    ) -> AskResult:
        logger.info("Agent question", question=question[:80], user_id=user_id)

        validate_question(question)

        has_doc_scope = bool(document_id) or bool(document_ids)

        system = SYSTEM_PROMPT
        if document_id:
            system += (
                f"\n\nIMPORTANT: The user has selected document '{document_id}'. "
                f"ALWAYS search only this document unless the user explicitly asks "
                f"about another document. Pass document_id='{document_id}' to every "
                f"search_documents call."
            )
        # elif document_ids:
        #     system += (
        #         f"\n\nIMPORTANT: The user has selected document '{document_ids}'. "
        #         f"ALWAYS search only this document unless the user explicitly asks "
        #         f"about another document. Pass document_id='{document_ids}' to every "
        #         f"search_documents call."
        #     )


        trimmed_history = history[-6:] if len(history) > 6 else history

        messages = [
            {"role": "system", "content": system},
            *trimmed_history,
            {"role": "user",   "content": question},
        ]

        tool_call_log: list[dict] = []
        all_chunks:    list[dict] = []
        iterations = 0

        while iterations < MAX_ITERATIONS:
            iterations += 1
            logger.info("Agent iteration", n=iterations)

            active_tools = (
                [t for t in TOOLS if t["function"]["name"] != "list_documents"]
                if has_doc_scope else TOOLS
            )

            tool_choice = (
                {"type": "function", "function": {"name": "search_documents"}}
                if has_doc_scope and iterations == 1
                else "auto"
            )

            response = await self._chat.complete(
                messages    = messages,
                tools       = TOOLS,
                tool_choice = tool_choice,
                max_tokens  = self._max_tokens,
            )

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_unset=True))

            if not msg.tool_calls:
                cited_pages = self._extract_cited_pages(
                    msg.content or "",
                    all_chunks,
                )
                logger.info("Agent done", iterations=iterations, pages=cited_pages)
                return AskResult(
                    answer       = msg.content or "",
                    source_pages = cited_pages,
                    query_type   = QueryType.ANALYTICAL,
                    tool_calls   = tool_call_log,
                )

            for tc in msg.tool_calls:
                fn   = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                logger.info("Tool call", tool=fn, args=args)

                tool_result, retrieved_chunks = await self._execute_tool(
                    fn, args, document_id, document_ids,
                    user_id = user_id,              # ← ADDED
                )
                all_chunks.extend(retrieved_chunks)
                tool_call_log.append({"tool": fn, "args": args})

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      tool_result,
                })

        logger.warning("Max iterations reached")
        return AskResult(
            answer       = "Could not produce an answer within the iteration limit.",
            source_pages = [],
            query_type   = QueryType.ANALYTICAL,
            tool_calls   = tool_call_log,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_cited_pages(
        self,
        answer:     str,
        all_chunks: list[dict],
    ) -> list[int]:
        cited = []

        patterns = [
            r'[Ss]ources?:\s*pages?\s*([\d,\s]+)',
            r'\bpages?\s+(\d+(?:\s*,\s*\d+)*)',
            r'\bpages?\s+(\d+)\s+and\s+(\d+)',
            r'\bp\.(\d+)\b',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, answer)
            for match in matches:
                if isinstance(match, tuple):
                    for m in match:
                        for num in re.findall(r'\d+', str(m)):
                            cited.append(int(num))
                else:
                    for num in re.findall(r'\d+', match):
                        cited.append(int(num))

        if cited:
            return sorted(set(cited))

        if all_chunks:
            best = max(all_chunks, key=lambda c: c.get("score", 0))
            p = best.get("page_number")
            if p is not None:
                return [p]

        return []

    # ── Tool router ───────────────────────────────────────────────────────────

    async def _execute_tool(
        self,
        fn:           str,
        args:         dict,
        document_id:  str | None,
        document_ids: list[str] | None,
        user_id:      str | None = None,        # ← ADDED
    ) -> tuple[str, list[dict]]:
        if fn == "search_documents":
            return await self._tool_search_documents(
                args, document_id, document_ids,
                user_id = user_id,              # ← ADDED
            )
        if fn == "search_emails":
            result = await self._tool_search_emails(
                args,
                user_id = user_id,              # ← ADDED
            )
            return result, []
        if fn == "list_documents":
            if document_id:
                args["query"] = "overview contents summary"
                return await self._tool_search_documents(
                    args, document_id, document_ids,
                    user_id = user_id,          # ← ADDED
                )
            result = await self._tool_list(user_id=user_id)  # ← ADDED
            return result, []
        return json.dumps({"error": f"Unknown tool: {fn}"}), []

    # ── Tool implementations ──────────────────────────────────────────────────

    @traceable(name="search_documents")
    async def _tool_search_documents(
        self,
        args:         dict,
        document_id:  str | None,
        document_ids: list[str] | None,
        user_id:      str | None = None,        # ← ADDED
    ) -> tuple[str, list[dict]]:
        query  = args.get("query", "")
        top_k  = min(int(args.get("top_k", self._top_k)), 20)
        doc_id = args.get("document_id") or document_id

        if isinstance(doc_id, list):
            if len(doc_id) == 1:
                doc_id = doc_id[0]
            else:
                document_ids = doc_id
                doc_id = None

        embeddings = await self._embedder.create_embeddings([query])
        if not embeddings:
            return json.dumps({"results": [], "message": "Embedding failed."}), []

        chunks = await self._store.hybrid_search(
            query        = query,
            embedding    = embeddings[0],
            top_k        = top_k,
            document_id  = doc_id,
            document_ids = document_ids if not doc_id else None,
            user_id      = user_id,             # ← ADDED
        )

        chunks = [
            c for c in chunks
            if len(c.get("content", "")) >= self._min_chunk_length
        ]

        chunk_embeddings = [c.get("embedding", []) for c in chunks]

        chunks = mmr_rerank(
            query_embedding  = embeddings[0],
            chunks           = chunks,
            chunk_embeddings = chunk_embeddings,
            top_k            = self._mmr_top_k,
            diversity        = self._mmr_diversity,
        )

        chunks = truncate_chunks_to_limit(chunks)

        results = [
            {
                "page":   c.get("page_number"),
                "text":   c.get("content", ""),
                "doc_id": c.get("document_id", ""),
                "score":  c.get("score", 0),
            }
            for c in chunks
        ]

        logger.info("search_documents", returned=len(results))
        return json.dumps({"results": results}), chunks

    async def _tool_search_emails(
        self,
        args:    dict,
        user_id: str | None = None,             # ← ADDED
    ) -> str:
        query = args.get("query", "")
        top_k = min(int(args.get("top_k", 10)), 20)

        embeddings = await self._embedder.create_embeddings([query])
        if not embeddings:
            return json.dumps({"results": [], "message": "Embedding failed."})

        results = await self._store.search_emails(
            query     = query,
            embedding = embeddings[0],
            top_k     = top_k,
            user_id   = user_id,                # ← ADDED
        )

        logger.info("search_emails", returned=len(results))
        return json.dumps({"results": results})

    async def _tool_list(
        self,
        user_id: str | None = None,             # ← ADDED
    ) -> str:
        docs = await self._store.list_documents(
            user_id = user_id,                  # ← ADDED
        )
        return json.dumps({"documents": docs})