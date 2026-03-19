"""
RAG Pipeline API — Document-grounded Q&A with citation support.

A production-ready Retrieval-Augmented Generation pipeline built with
FastAPI, LangChain, and PostgreSQL + pgvector.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, health
from app.config import settings
from app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and clean up on shutdown."""
    await init_db()
    yield


app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "Upload documents, chunk & embed them with pgvector, "
        "and ask questions with cited answers powered by LangChain."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
