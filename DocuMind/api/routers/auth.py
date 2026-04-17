from __future__ import annotations

import aiosqlite
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

from DocuMind.core.auth.database import get_db
from DocuMind.core.auth.service import register_user, login_user, get_user_by_id
from DocuMind.core.auth.jwt_handler import create_token, verify_token
from DocuMind.core.logging.logger import get_logger

logger = get_logger(__name__)
router  = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer()


# ── Request / Response models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email:    EmailStr
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token:    str
    user_id:  int
    username: str


# ── Dependency — use in any protected route ───────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db:          aiosqlite.Connection = Depends(get_db),
) -> dict:
    """
    FastAPI dependency that:
    1. Reads Bearer token from Authorization header
    2. Verifies it is valid and not expired
    3. Returns the user dict

    Any route that needs auth adds:
        user = Depends(get_current_user)
    and gets the logged-in user automatically.
    """
    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = await get_user_by_id(db, payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse)
async def register(
    request: RegisterRequest,
    db:      aiosqlite.Connection = Depends(get_db),
):
    """
    Register a new user.
    Returns a JWT token immediately — no need to login after register.
    """
    if len(request.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters"
        )

    user = await register_user(
        db       = db,
        username = request.username,
        email    = request.email,
        password = request.password,
    )

    if not user:
        raise HTTPException(
            status_code=400,
            detail="Username or email already taken"
        )

    token = create_token(user["id"], user["username"])
    return AuthResponse(token=token, user_id=user["id"], username=user["username"])


@router.post("/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    db:      aiosqlite.Connection = Depends(get_db),
):
    """
    Login with username and password.
    Returns a JWT token valid for 30 days.
    """
    user = await login_user(
        db       = db,
        username = request.username,
        password = request.password,
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    token = create_token(user["id"], user["username"])
    return AuthResponse(token=token, user_id=user["id"], username=user["username"])


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    """
    Returns current logged-in user info.
    Frontend can call this to verify token is still valid.
    """
    return user