from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from DocuMind.core.logging.logger import setup_logging, get_logger
from DocuMind.core.settings import get_settings
from DocuMind.api.routers.documents import router as documents_router
from DocuMind.api.routers.email import router as email_router
from DocuMind.search.azure_search_store import AzureSearchStore

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan — runs at startup and shutdown.
    Creates Azure Search index on startup if it doesn't exist.
    """
    settings = get_settings()
    setup_logging(
        log_level      = settings.log_level,
        is_development = settings.is_development,
    )

    # Initialize Azure Search index
    store = AzureSearchStore()
    await store.ensure_index()
    app.state.store = store
    logger.info("DocuMind API started")

    yield

    logger.info("DocuMind API stopped")


app = FastAPI(
    title       = "DocuMind API",
    description = "AI-powered document intelligence",
    version     = "1.0.0",
    lifespan    = lifespan,
)

origins = ["*"]
# CORS — allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins     = origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Register routers
app.include_router(documents_router)
app.include_router(email_router)