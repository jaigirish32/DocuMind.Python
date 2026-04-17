from __future__ import annotations

import aiosqlite
from pathlib import Path
from DocuMind.core.logging.logger import get_logger

logger = get_logger(__name__)

# Database file location — sits in project root
DB_PATH = Path("documind.db")


async def init_db() -> None:
    """
    Create the database and users table if they don't exist.
    Called once on server startup.
    
    aiosqlite is async SQLite — works with FastAPI's async model.
    Regular sqlite3 would BLOCK the event loop during DB operations.
    aiosqlite runs DB operations in a thread pool — non-blocking.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    UNIQUE NOT NULL,
                email         TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    DEFAULT (datetime('now'))
            )
        """)
        await db.commit()
        logger.info("Database initialized", path=str(DB_PATH))


async def get_db() -> aiosqlite.Connection:
    """
    Get a database connection.
    Used as a FastAPI dependency in route handlers.
    
    Usage in routes:
        async def my_route(db = Depends(get_db)):
            ...
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row  # rows behave like dicts
        yield db