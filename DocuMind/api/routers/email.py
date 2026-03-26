from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from DocuMind.core.logging.logger import get_logger
from DocuMind.email.providers.gmail import GmailProvider
from DocuMind.email.indexer import EmailIndexer
from DocuMind.api.dependencies import get_embedding_client

logger = get_logger(__name__)
router = APIRouter(prefix="/api/email", tags=["email"])


# ── Request / Response models ─────────────────────────────────────────────────

class SyncRequest(BaseModel):
    max_results: int         = 50
    query:       str         = ""
    label:       str         = "INBOX"


class SyncResponse(BaseModel):
    emails:  int
    chunks:  int
    message: str


class SearchRequest(BaseModel):
    question:    str
    max_results: int = 10


class SearchResponse(BaseModel):
    results: list[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/sync/gmail", response_model=SyncResponse)
async def sync_gmail(request: Request, body: SyncRequest):
    """
    Fetch emails from Gmail and index them into Weaviate.
    First call opens browser for Google OAuth consent.
    Subsequent calls are silent — token auto refreshes.
    """
    try:
        # Fetch emails from Gmail
        provider = GmailProvider()
        emails   = provider.fetch_emails(
            max_results = body.max_results,
            query       = body.query,
            label       = body.label,
        )

        if not emails:
            return SyncResponse(
                emails  = 0,
                chunks  = 0,
                message = "No emails found",
            )

        # Index into Weaviate
        store   = request.app.state.store
        indexer = EmailIndexer(
            embedder = get_embedding_client(),
            store    = store,
        )
        result = await indexer.index(emails)

        return SyncResponse(
            emails  = result["emails"],
            chunks  = result["chunks"],
            message = f"Successfully indexed {result['emails']} emails ({result['chunks']} chunks)",
        )

    except Exception as e:
        logger.error("Gmail sync failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search_emails(request: Request, body: SearchRequest):
    """
    Search indexed emails using hybrid search.
    """
    try:
        embedder   = get_embedding_client()
        embeddings = await embedder.create_embeddings([body.question])

        if not embeddings:
            raise HTTPException(status_code=500, detail="Embedding failed")

        store   = request.app.state.store
        results = await store.search_emails(
            query     = body.question,
            embedding = embeddings[0],
            top_k     = body.max_results,
        )

        return SearchResponse(results=results)

    except Exception as e:
        logger.error("Email search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def email_status(request: Request):
    """Check how many emails are indexed."""
    try:
        store  = request.app.state.store
        count  = await store.count_emails()
        return {"indexed_emails": count}
    except Exception as e:
        return {"indexed_emails": 0, "error": str(e)}