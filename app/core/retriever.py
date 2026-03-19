"""Retrieve relevant document chunks via pgvector similarity search."""

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.embeddings import embed_query
from app.db.vector_store import Document, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned from similarity search with its score."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    content: str
    page_number: int | None
    similarity_score: float


async def retrieve(
    session: AsyncSession,
    query: str,
    *,
    document_ids: list[uuid.UUID] | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[RetrievedChunk]:
    """
    Perform cosine similarity search against stored chunk embeddings.

    Args:
        session: Active database session.
        query: The user's question.
        document_ids: Optional list of document IDs to scope the search.
        top_k: Number of results to return.
        threshold: Minimum similarity score (0-1).

    Returns:
        List of RetrievedChunk ordered by descending similarity.
    """
    k = top_k or settings.RETRIEVAL_TOP_K
    min_score = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

    query_vector = await embed_query(query)

    # Build the similarity search query using pgvector's <=> (cosine distance)
    # cosine_similarity = 1 - cosine_distance
    similarity_expr = (1 - DocumentChunk.embedding.cosine_distance(query_vector)).label(
        "similarity"
    )

    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            DocumentChunk.page_number,
            Document.filename,
            similarity_expr,
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(text("similarity DESC"))
        .limit(k)
    )

    if min_score > 0:
        stmt = stmt.where(
            (1 - DocumentChunk.embedding.cosine_distance(query_vector)) >= min_score
        )

    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))

    result = await session.execute(stmt)
    rows = result.all()

    chunks = [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            filename=row.filename,
            chunk_index=row.chunk_index,
            content=row.content,
            page_number=row.page_number,
            similarity_score=round(float(row.similarity), 4),
        )
        for row in rows
    ]

    logger.info(
        "Retrieved %d chunks (top_k=%d, threshold=%.2f) for query: %s",
        len(chunks),
        k,
        min_score,
        query[:80],
    )
    return chunks
