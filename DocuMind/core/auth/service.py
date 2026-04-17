from __future__ import annotations

import bcrypt
import aiosqlite
from DocuMind.core.logging.logger import get_logger

logger = get_logger(__name__)


async def register_user(
    db:       aiosqlite.Connection,
    username: str,
    email:    str,
    password: str,
) -> dict | None:
    """
    Register a new user.
    
    Steps:
    1. Hash the password with bcrypt
       bcrypt is slow by design — makes brute force attacks harder
       Each hash includes a random salt — same password → different hash
    2. Insert into users table
    3. Return user dict or None if username/email taken
    
    We never store the plain password.
    Even if our database is stolen,
    attacker can't reverse the hash to get passwords.
    """
    # Hash password — bcrypt adds salt automatically
    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=12),   # 12 = work factor (higher = slower = safer)
    ).decode("utf-8")

    try:
        cursor = await db.execute(
            """
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (username.lower().strip(), email.lower().strip(), password_hash),
        )
        await db.commit()

        user_id = cursor.lastrowid
        logger.info("User registered", username=username, user_id=user_id)

        return {
            "id":       user_id,
            "username": username,
            "email":    email,
        }

    except aiosqlite.IntegrityError:
        # Username or email already exists
        logger.warning("Registration failed — duplicate", username=username)
        return None


async def login_user(
    db:       aiosqlite.Connection,
    username: str,
    password: str,
) -> dict | None:
    """
    Verify login credentials.
    
    Steps:
    1. Find user by username in database
    2. Compare provided password against stored hash
       bcrypt.checkpw does this safely
    3. Return user dict if valid, None if invalid
    
    We never compare plain passwords directly.
    bcrypt.checkpw handles the comparison correctly.
    """
    cursor = await db.execute(
        "SELECT id, username, email, password_hash FROM users WHERE username = ?",
        (username.lower().strip(),),
    )
    row = await cursor.fetchone()

    if not row:
        logger.warning("Login failed — user not found", username=username)
        return None

    # Check password against stored hash
    password_matches = bcrypt.checkpw(
        password.encode("utf-8"),
        row["password_hash"].encode("utf-8"),
    )

    if not password_matches:
        logger.warning("Login failed — wrong password", username=username)
        return None

    logger.info("User logged in", username=username, user_id=row["id"])
    return {
        "id":       row["id"],
        "username": row["username"],
        "email":    row["email"],
    }


async def get_user_by_id(
    db:      aiosqlite.Connection,
    user_id: int,
) -> dict | None:
    """Get user by ID — used to verify token is still valid."""
    cursor = await db.execute(
        "SELECT id, username, email FROM users WHERE id = ?",
        (user_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return {"id": row["id"], "username": row["username"], "email": row["email"]}