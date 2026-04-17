from __future__ import annotations

from jose import jwt
from datetime import datetime, timedelta, timezone
from DocuMind.core.settings import get_settings
from DocuMind.core.logging.logger import get_logger

logger = get_logger(__name__)

# Token expires after 30 days
TOKEN_EXPIRE_DAYS = 30


def create_token(user_id: int, username: str) -> str:
    """
    Create a JWT token for a user.
    
    JWT has three parts:
    1. Header  — algorithm used (HS256)
    2. Payload — user_id, username, expiry time
    3. Signature — proves token is genuine
    
    Only our server can create valid signatures
    because only we know the SECRET_KEY.
    """
    settings = get_settings()

    payload = {
        "user_id":  user_id,
        "username": username,
        "exp":      datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS),
        "iat":      datetime.now(timezone.utc),  # issued at
    }

    token = jwt.encode(
        payload,
        settings.secret_key,
        algorithm="HS256",
    )
    return token


def verify_token(token: str) -> dict | None:
    """
    Verify a JWT token and return its payload.
    
    Returns None if:
    - Token was tampered with
    - Token has expired
    - Token is malformed
    
    Returns dict with user_id and username if valid.
    """
    settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=["HS256"],
        )
        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None

    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        return None