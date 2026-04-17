from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from DocuMind.api.limiter import limiter

from DocuMind.core.logging.logger import setup_logging, get_logger
from DocuMind.core.settings import get_settings
from DocuMind.api.routers.documents import router as documents_router
from DocuMind.api.routers.email import router as email_router
from DocuMind.search.azure_search_store import AzureSearchStore
from DocuMind.core.auth.database import init_db
from DocuMind.api.routers.auth import router as auth_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(
        log_level      = settings.log_level,
        is_development = settings.is_development,
    )
    await init_db()
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

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.options("/{path:path}")
async def options_handler(path: str, request: Request):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(email_router)