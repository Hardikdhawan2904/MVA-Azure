import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.database import init_db
from app.routes.upload import router as upload_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize database on startup."""
    logger.info("Starting Dataset Identification Agent...")
    init_db()
    logger.info("Database initialized. Agent is ready.")
    yield
    logger.info("Shutting down Dataset Identification Agent.")


app = FastAPI(
    title="Dataset Identification Agent",
    description=(
        "Ingests CSV and Excel datasets, extracts metadata, classifies the "
        "business domain using an LLM (Azure OpenAI), and prepares data for downstream agents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(upload_router, tags=["Dataset Operations"])


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Dataset Identification Agent"}
