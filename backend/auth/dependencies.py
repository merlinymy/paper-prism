"""FastAPI dependencies for authentication.

Auth can be disabled entirely by setting ENABLE_AUTH=false in .env.
When disabled, all endpoints use the default user — no login required.
"""

import logging
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import get_async_session, User
from .auth import decode_access_token

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)


def _is_auth_enabled() -> bool:
    """Check if authentication is enabled via settings."""
    return settings.enable_auth


async def _get_default_user(session: AsyncSession) -> User:
    """Get the default user for auth-disabled mode."""
    result = await session.execute(
        select(User).where(User.username == settings.default_username)
    )
    user = result.scalar_one_or_none()
    if user is None:
        # This shouldn't happen — default user is created on startup
        raise HTTPException(
            status_code=500,
            detail="Default user not found. Database may not be initialized.",
        )
    return user


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security_optional)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> User:
    """Get the current authenticated user.

    When ENABLE_AUTH=false, returns the default user without checking tokens.
    When ENABLE_AUTH=true, requires a valid JWT token.
    """
    # Auth disabled — return default user, no token needed
    if not _is_auth_enabled():
        return await _get_default_user(session)

    # Auth enabled — require valid token
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None:
        raise credentials_exception

    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

    result = await session.execute(
        select(User).where(User.username == username)
    )
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    return user


async def get_current_user_optional(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security_optional)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> Optional[User]:
    """Get the current user, or None if not authenticated.

    When ENABLE_AUTH=false, always returns the default user.
    When ENABLE_AUTH=true, returns None if no valid token.
    """
    # Auth disabled — always return default user
    if not _is_auth_enabled():
        return await _get_default_user(session)

    if credentials is None:
        return None

    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None:
        return None

    username: str = payload.get("sub")
    if username is None:
        return None

    result = await session.execute(
        select(User).where(User.username == username)
    )
    return result.scalar_one_or_none()
