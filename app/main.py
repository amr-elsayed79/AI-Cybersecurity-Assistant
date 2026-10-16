"""FastAPI application entry point for the AI Cybersecurity Assistant.

DEPLOYMENT & SECURITY NOTICE:
This API is designed strictly for local single-process demonstration and evaluation.
- Binding: Must remain bound to 127.0.0.1 (localhost). Do NOT bind to 0.0.0.0 or expose publicly.
- Memory: In-memory conversation state is volatile and not shared across multi-process workers.
- Access Control: Session IDs provide unguessability for local demos, not authenticated multi-tenant isolation.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.chat import chat_endpoint, router as chat_router
from app.api.v1.health import get_health, router as health_router
from app.config import settings
from app.rag.retriever import close_shared_vector_store

# Resolve frontend directory path relative to project location
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Configure standard logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and graceful shutdown."""
    yield
    # Safely close shared vector store and release filesystem lock on shutdown
    close_shared_vector_store()


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "AI Cybersecurity Assistant — Grounded RAG chat API with strict source attribution "
            "and deterministic guardrails. Intended for local evaluation bound to 127.0.0.1."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Global Exception Handlers for Security and Hygiene
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle request schema validation errors cleanly without exposing internals."""
        errors = []
        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "Validation error")
            errors.append(f"{loc}: {msg}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Validation error", "errors": errors},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Pass through HTTPExceptions with clean detail messages."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all for unhandled exceptions to prevent stack trace or path disclosure."""
        logger.error("Unhandled server exception: %s", type(exc).__name__, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred."},
        )

    # Root informational endpoint
    @app.get("/", tags=["Info"])
    def root_info() -> Dict[str, Any]:
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "running",
            "health_endpoint": "/health",
            "chat_endpoint": f"{settings.API_V1_PREFIX}/chat",
            "dashboard_endpoint": "/dashboard/",
        }

    # Mount health endpoint at root /health and /api/v1/health
    app.add_api_route(
        "/health",
        get_health,
        methods=["GET"],
        tags=["Health"],
        summary="Root health check",
    )
    app.include_router(health_router, prefix=settings.API_V1_PREFIX)

    # Mount chat router at /api/v1 and alias at /api
    app.include_router(chat_router, prefix=settings.API_V1_PREFIX)
    app.add_api_route(
        "/api/chat",
        chat_endpoint,
        methods=["POST"],
        tags=["Chat"],
        summary="Chat alias endpoint",
    )

    # Mount static frontend demo dashboard at /dashboard
    if FRONTEND_DIR.exists() and FRONTEND_DIR.is_dir():
        app.mount("/dashboard", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="dashboard")

    return app


# Default application instance for Uvicorn
app = create_app()
