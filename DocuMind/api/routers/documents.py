from __future__ import annotations

import uuid
from pathlib import Path

import tempfile
import os

import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel

from DocuMind.core.logging.logger import get_logger
from DocuMind.documents.readers.pdf_reader import PdfReader
from DocuMind.documents.indexing.document_indexer import DocumentIndexer
from DocuMind.api.dependencies import get_embedding_client, get_chat_client
from DocuMind.agents.documind_agent import DocuMindAgent

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["documents"])


# ── Request / Response models ─────────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id:   str
    document_name: str
    pages:         int
    chunks:        int
    message:       str


class AskRequest(BaseModel):
    question:    str
    document_id: str | None = None


class AskResponse(BaseModel):
    answer:       str
    source_pages: list[int]
    document_id:  str | None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file:    UploadFile = File(...),
):
    """Upload a PDF and index it in Weaviate."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code = 400,
            detail      = "Only PDF files are supported"
        )

    document_id = str(uuid.uuid4())
    tmp_dir     = tempfile.gettempdir()
    tmp_path    = Path(tmp_dir) / file.filename

    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        store    = request.app.state.store
        embedder = get_embedding_client()
        reader   = PdfReader()

        indexer = DocumentIndexer(
            reader   = reader,
            embedder = embedder,
            store    = store,
        )

        result = await indexer.index(
            path=tmp_path,
            document_id = document_id,
        )

        return UploadResponse(
            document_id   = document_id,
            document_name = file.filename,
            pages         = result["pages"],
            chunks        = result["chunks"],
            message       = "Document indexed successfully",
        )

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: Request, body: AskRequest):
    """Ask a question — optionally filter by document_id."""
    if not body.question.strip():
        raise HTTPException(
            status_code = 400,
            detail      = "Question cannot be empty"
        )

    store = request.app.state.store
    agent = DocuMindAgent(
        embedder = get_embedding_client(),
        chat     = get_chat_client(),
        store    = store,
    )

    result = await agent.ask_structured(
        question    = body.question,
        document_id = body.document_id,
    )

    return AskResponse(
        answer       = result.answer,
        source_pages = result.source_pages,
        document_id  = body.document_id,
    )


@router.get("/documents")
async def list_documents(request: Request):
    """List all indexed documents."""
    store = request.app.state.store
    docs  = await store.list_documents()
    return {"documents": docs}


@router.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}