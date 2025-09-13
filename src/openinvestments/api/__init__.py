"""
FastAPI application for OpenInvestments platform.

Provides REST API endpoints for:
- Option pricing and valuation
- Risk calculations (VaR, ES)
- Portfolio analysis
- Model validation
- Reporting and data export
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager

from ..core.config import config
from ..core.logging import setup_logging, get_logger
from ..observability.middleware import setup_observability
from .routes import valuation, risk, portfolio, validation, reports

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting OpenInvestments API server")
    yield
    # Shutdown
    logger.info("Shutting down OpenInvestments API server")


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""

    # Set up logging
    setup_logging()

    # Create FastAPI app
    app = FastAPI(
        title="OpenInvestments Quantitative Risk API",
        description="REST API for quantitative risk analytics and derivatives pricing",
        version=config.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )

    # Set up observability
    setup_observability(app)

    # Add authentication middleware (permissive by default for smoke/local use)
    try:
        from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
        from .security import verify_token
        security = HTTPBearer(auto_error=False)

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            path = request.url.path
            open_paths = {"/", "/health", "/openapi.json"}
            if path in open_paths or path.startswith("/docs"):
                return await call_next(request)
            credentials: HTTPAuthorizationCredentials = await security(request)
            token = credentials.credentials if credentials else None
            if not verify_token(token):
                raise HTTPException(status_code=401, detail="Invalid token")
            return await call_next(request)
    except Exception:
        # If security modules unavailable, skip auth
        pass

    # Include routers
    app.include_router(
        valuation.router,
        prefix="/api/v1/valuation",
        tags=["valuation"]
    )

    app.include_router(
        risk.router,
        prefix="/api/v1/risk",
        tags=["risk"]
    )

    app.include_router(
        portfolio.router,
        prefix="/api/v1/portfolio",
        tags=["portfolio"]
    )

    app.include_router(
        validation.router,
        prefix="/api/v1/validation",
        tags=["validation"]
    )

    app.include_router(
        reports.router,
        prefix="/api/v1/reports",
        tags=["reports"]
    )

    from datetime import datetime

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": config.version,
            "timestamp": datetime.utcnow().isoformat()
        }

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "OpenInvestments Quantitative Risk Platform API",
            "version": config.version,
            "docs": "/docs",
            "health": "/health"
        }

    logger.info("FastAPI application created successfully")
    return app


# Create the application instance
app = create_application()
