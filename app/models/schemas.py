"""Pydantic schemas for request and response payloads."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────── Documents ────────────────────────────


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    content_type: str
    total_chunks: int
    file_size_bytes: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListOut(BaseModel):
    total: int
    documents: list[DocumentOut]


class IngestionResult(BaseModel):
    document_id: uuid.UUID
    filename: str
    total_chunks: int
    message: str = "Document ingested successfully"


# ──────────────────────────── Chat / Q&A ───────────────────────────


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        json_schema_extra={"example": "What is this document about?"},
    )
    document_ids: Optional[list[uuid.UUID]] = Field(
        default=None,
        description="Scope retrieval to specific documents. If None, search all.",
    )
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is this document about?",
                    "document_ids": None,
                    "top_k": 5,
                }
            ]
        }
    }


class Citation(BaseModel):
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    content_preview: str = Field(
        ..., description="First 300 chars of the retrieved chunk"
    )
    page_number: Optional[int] = None
    similarity_score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model: str
    usage: Optional[dict] = None


# ──────────────────────────── Health ───────────────────────────────


class HealthOut(BaseModel):
    status: str = "ok"
    version: str
    database: str = "connected"
