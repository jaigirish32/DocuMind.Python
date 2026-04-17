from __future__ import annotations

from pathlib import Path
import tempfile
import re
import aiofiles
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Request, Form
from pydantic import BaseModel

from DocuMind.core.settings import get_settings
from DocuMind.core.logging.logger import get_logger
from DocuMind.documents.indexing.document_indexer import DocumentIndexer
from DocuMind.api.dependencies import get_embedding_client, get_chat_client
from DocuMind.agents.documind_agent import DocuMindAgent
from DocuMind.documents.readers.azure_document_reader import AzureDocumentReader
from DocuMind.api.limiter import limiter
from DocuMind.api.routers.auth import get_current_user

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["documents"])


# ── Request / Response models ─────────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id:   str
    document_name: str
    pages:         int
    chunks:        int
    message:       str
    category:      str = "Others"


class AskRequest(BaseModel):
    question:     str
    document_id:  str | None = None
    document_ids: list[str] | None = None
    history:      list[dict] = []


class AskResponse(BaseModel):
    answer:       str
    source_pages: list[int]
    document_id:  str | None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
@limiter.limit("10/hour")
async def upload_document(
    request:  Request,
    file:     UploadFile = File(...),
    category: str = Form("Others"),
    user:     dict = Depends(get_current_user),
):
    """Upload a PDF and index it in Azure Search."""
    logger.info("Upload request", category=category, filename=file.filename)
    user_id = str(user["id"])

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    settings = get_settings()
    if settings.max_upload_size_mb > 0:
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"File too large. Maximum size is {settings.max_upload_size_mb} MB. "
                    f"Your file is {size_mb:.1f} MB."
                ),
            )
    await file.seek(0)

    document_id = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(file.filename).stem)
    tmp_path    = Path(tempfile.gettempdir()) / file.filename

    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        indexer = DocumentIndexer(
            reader   = AzureDocumentReader(),
            embedder = get_embedding_client(),
            store    = request.app.state.store,
        )

        result = await indexer.index(
            path        = tmp_path,
            document_id = document_id,
            category    = category,
            user_id     = user_id,
        )

        return UploadResponse(
            document_id   = document_id,
            document_name = file.filename,
            pages         = result["pages"],
            chunks        = result["chunks"],
            message       = "Document indexed successfully",
            category      = category,
        )

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
async def ask_question(
    request: Request,
    body:    AskRequest,
    user:    dict = Depends(get_current_user),
):
    """Ask a question about indexed documents."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    user_id = str(user["id"])

    agent = DocuMindAgent(
        embedder = get_embedding_client(),
        chat     = get_chat_client(),
        store    = request.app.state.store,
    )

    result = await agent.ask_structured(
        question     = body.question,
        document_id  = body.document_id,
        document_ids = body.document_ids,
        history      = body.history,
        user_id      = user_id,
    )

    return AskResponse(
        answer       = result.answer,
        source_pages = result.source_pages,
        document_id  = body.document_id,
    )


@router.get("/documents")
async def list_documents(
    request: Request,
    user:    dict = Depends(get_current_user),
):
    """List all documents for current user."""
    user_id = str(user["id"])
    docs    = await request.app.state.store.list_documents(user_id=user_id)
    return {"documents": docs}


@router.get("/health")
async def health():
    """Health check — no auth needed."""
    return {"status": "ok"}