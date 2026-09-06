"""FastAPI routes for QQ email verification and cookie sessions."""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from .service import (
    AuthError,
    AuthService,
    EmailConfigurationError,
    EmailDeliveryError,
    EmailValidationError,
    ExpiredVerificationCodeError,
    InvalidSessionError,
    InvalidVerificationCodeError,
    VerificationCooldownError,
)


LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class VerificationCodeRequest(BaseModel):
    email: str = Field(min_length=1, max_length=254)


class VerificationCodeLogin(BaseModel):
    email: str = Field(min_length=1, max_length=254)
    code: str = Field(min_length=6, max_length=6)


@lru_cache(maxsize=1)
def get_auth_service() -> AuthService:
    """Reuse one configured authentication service for the process."""
    return AuthService()


def _http_error(exc: AuthError) -> HTTPException:
    if isinstance(exc, EmailValidationError):
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))
    if isinstance(exc, VerificationCooldownError):
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )
    if isinstance(exc, EmailConfigurationError):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    if isinstance(exc, EmailDeliveryError):
        return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    if isinstance(exc, (InvalidVerificationCodeError, ExpiredVerificationCodeError)):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if isinstance(exc, InvalidSessionError):
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Session"},
        )
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


def get_optional_current_user(
    request: Request,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> dict[str, object] | None:
    """Resolve a session when present without rejecting public pages."""
    session_token = request.cookies.get(auth_service.settings.session_cookie_name)
    try:
        return auth_service.get_user_by_session(session_token)
    except InvalidSessionError:
        return None


def require_current_user(
    current_user: Annotated[
        dict[str, object] | None,
        Depends(get_optional_current_user),
    ],
) -> dict[str, object]:
    """Require a valid session for protected API endpoints."""
    if current_user is None:
        exc = InvalidSessionError("未登录")
        raise _http_error(exc) from exc
    return current_user


@router.post("/code/request", status_code=status.HTTP_202_ACCEPTED)
def request_verification_code(
    payload: VerificationCodeRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> dict[str, object]:
    """Send a short-lived login code without exposing it in the response."""
    try:
        result = auth_service.request_verification_code(payload.email)
    except AuthError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        LOGGER.exception("[Auth] verification_code_send_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="验证码服务暂时不可用",
        ) from exc
    return {"message": "验证码已发送", **result}


@router.post("/code/verify")
def verify_code_and_login(
    payload: VerificationCodeLogin,
    response: Response,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> dict[str, object]:
    """Verify the code, register when necessary, and establish a session."""
    try:
        result = auth_service.verify_and_login(payload.email, payload.code)
    except AuthError as exc:
        raise _http_error(exc) from exc

    settings = auth_service.settings
    response.set_cookie(
        key=settings.session_cookie_name,
        value=result.session_token,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        path="/",
    )
    return {"authenticated": True, "user": result.user}


@router.get("/me")
def current_user(
    user: Annotated[dict[str, object], Depends(require_current_user)],
) -> dict[str, object]:
    return {"authenticated": True, "user": user}


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> dict[str, object]:
    session_token = request.cookies.get(auth_service.settings.session_cookie_name)
    auth_service.logout(session_token)
    response.delete_cookie(
        key=auth_service.settings.session_cookie_name,
        path="/",
        secure=auth_service.settings.session_cookie_secure,
        httponly=True,
        samesite="lax",
    )
    return {"authenticated": False}


__all__ = [
    "get_auth_service",
    "get_optional_current_user",
    "require_current_user",
    "router",
]
