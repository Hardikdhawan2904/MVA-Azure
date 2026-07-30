"""Authentication middleware — supports API key and JWT."""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import Settings
from app.api.dependencies import get_cached_settings

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# JWT Bearer token
bearer_scheme = HTTPBearer(auto_error=False)


def verify_api_key(
    api_key: str | None = Security(api_key_header),
    settings: Settings = Depends(get_cached_settings),
) -> str | None:
    """Verify API key if authentication is enabled."""
    configured_keys = settings.api_keys
    if not configured_keys:
        # No keys configured = auth disabled
        return None

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "MISSING_API_KEY", "message": "X-API-Key header is required."},
        )

    if api_key not in configured_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "INVALID_API_KEY", "message": "The provided API key is not valid."},
        )

    return api_key


def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    settings: Settings = Depends(get_cached_settings),
) -> str | None:
    """Verify JWT bearer token if configured."""
    jwt_secret = settings.jwt_secret
    if not jwt_secret:
        return None

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "MISSING_TOKEN", "message": "Bearer token is required."},
        )

    # Verify JWT
    try:
        import jwt as pyjwt
        payload = pyjwt.decode(
            credentials.credentials,
            jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        return payload.get("sub", "anonymous")
    except ImportError:
        # PyJWT not installed — skip JWT validation
        return credentials.credentials
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_TOKEN", "message": "The provided token is invalid or expired."},
        )


def get_current_user(
    api_key: str | None = Depends(verify_api_key),
    token_user: str | None = Depends(verify_bearer_token),
) -> str:
    """
    Resolve the authenticated user.

    Authentication is checked in order:
    1. API Key (X-API-Key header)
    2. Bearer token (Authorization header)
    3. If neither is configured, allows anonymous access

    Returns a user identifier string.
    """
    if api_key:
        return f"apikey:{api_key[:8]}..."
    if token_user:
        return token_user
    return "anonymous"
