"""FastAPI application entry point for the Local AgentOps backend."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.routes import demo_page, router as api_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.generation.service import GenerationService
from app.orchestration.service import OrchestrationService


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown side effects.

    This lifespan hook configures structured logging before the API starts
    accepting requests and emits lightweight lifecycle events that are useful
    during local development and future observability work.
    """
    # Load settings once during startup so runtime components can share a
    # consistent environment-backed configuration object.
    settings: Settings = get_settings()

    # Configure the root logging system before any application loggers are used.
    configure_logging(log_level=settings.log_level)

    # Create the managed data directories during startup so later ingestion,
    # indexing, and evaluation code can assume the expected filesystem layout
    # already exists.
    settings.create_data_directories()

    # Emit a startup event that records the active environment and API identity.
    logger = get_logger(__name__)
    logger.info("Application startup", extra={"environment": settings.environment})

    # Yield control back to FastAPI so request handling can begin.
    yield

    # Emit a matching shutdown event to make local runs easier to trace.
    logger.info("Application shutdown", extra={"environment": settings.environment})


def create_application() -> FastAPI:
    """Build and configure the FastAPI application instance.

    The function centralizes app creation so tests, scripts, and ASGI servers
    all use the same router registration and metadata setup.
    """
    # Read application settings to populate API metadata and runtime behavior.
    settings: Settings = get_settings()

    # Construct the FastAPI object with the shared lifespan manager.
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
        version="0.1.0",
        summary="Local-first AI Knowledge Ops API scaffold.",
    )

    # Store shared service instances on the application state so route handlers
    # and tests can reuse or override them without global singletons.
    application.state.orchestration_service = OrchestrationService()
    application.state.generation_service = GenerationService(settings=settings)

    # Attach the versioned API router so HTTP endpoints are grouped cleanly.
    application.include_router(api_router, prefix="/api")

    # Serve the minimal demo page at the root path so local users can inspect
    # grounded retrieval and answer generation without a separate frontend app.
    application.add_api_route("/", demo_page, methods=["GET"], response_class=HTMLResponse)

    # Return the configured app object to the caller.
    return application


# Create a module-level ASGI application so uvicorn can import it directly.
app = create_application()
