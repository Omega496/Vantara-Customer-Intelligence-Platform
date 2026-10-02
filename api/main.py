"""FastAPI Application entry point for Vantara Customer Intelligence Platform."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers.customers import router as customers_router
from api.routers.health import router as health_router
from api.routers.models import router as models_router
from api.routers.predict import router as predict_router
from api.services import ModelService
from src.database.connection import get_engine, init_db
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager to initialize DB and pre-warm model caches."""
    logger.info("Initializing Vantara Customer Intelligence API service...")
    try:
        engine = get_engine()
        init_db(engine)
        logger.info("Database connection and schema verified.")
    except Exception as e:
        logger.error(f"Error initializing database during startup: {e}")

    try:
        # Pre-warm ModelService in memory for sub-40ms initial requests
        svc = ModelService.get_instance()
        logger.info(f"Model service ready with {len(svc.feature_names)} features.")
    except Exception as e:
        logger.error(f"Error loading models during startup: {e}")

    yield

    logger.info("Shutting down Vantara Customer Intelligence API service.")


app = FastAPI(
    title="Vantara Customer Intelligence Platform API",
    version="1.0.0",
    description=(
        "Production-grade REST API providing end-to-end customer behavioral intelligence: "
        "90-day churn prediction, forward customer lifetime value (CLV) regression, "
        "unsupervised spending anomaly detection, behavioral customer segmentation, "
        "and automated plain-language marketing explanations."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit frontend and local dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health_router)
app.include_router(models_router)
app.include_router(predict_router)
app.include_router(customers_router)


@app.get("/", tags=["Root"])
def root_endpoint() -> JSONResponse:
    """Root endpoint welcoming clients and directing to OpenAPI documentation."""
    return JSONResponse(
        content={
            "platform": "Vantara Customer Intelligence Platform API",
            "version": "1.0.0",
            "status": "online",
            "docs": "/docs",
            "health": "/health",
            "models_metadata": "/models/metadata",
        }
    )
