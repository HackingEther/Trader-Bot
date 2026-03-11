"""Admin token authentication middleware."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

PROTECTED_PATHS = {"/api/killswitch", "/api/config"}


class AdminTokenMiddleware(BaseHTTPMiddleware):
    """Middleware protecting dangerous endpoints with admin token."""

    def __init__(self, app: object, admin_token: str) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._token = admin_token

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if any(path.startswith(p) for p in PROTECTED_PATHS):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != self._token:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Admin token required"},
                )
        return await call_next(request)
