"""Document upload, listing, and deletion endpoints."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.chunking import ChunkStrategy, chunk_pages
from app.core.embeddings import embed_texts
from app.core.ingestion import UnsupportedFileType, extract_text, save_upload
from app.db.database import get_session
from app.db.vector_store import Document, DocumentChunk
from app.models.schemas import DocumentListOut, DocumentOut, IngestionResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/documents",
    response_model=IngestionResult,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
)
async def upload_document(
    file: UploadFile,
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    session: AsyncSession = Depends(get_session),
):
    """
    Upload a document (PDF, DOCX, or TXT), extract text, chunk it,
    generate embeddings, and store everything in PostgreSQL + pgvector.
    """
    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit.",
        )

    # Save and extract text
    try:
        file_path = await save_upload(contents, file.filename)
        pages = await extract_text(file_path, file.content_type)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc))

    if not pages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the uploaded file.",
        )

    # Chunk
    chunks = chunk_pages(pages, strategy=strategy)

    # Embed
    texts = [c["content"] for c in chunks]
    vectors = await embed_texts(texts)

    # Persist
    doc = Document(
        filename=file.filename,
        content_type=file.content_type,
        total_chunks=len(chunks),
        file_size_bytes=len(contents),
    )
    session.add(doc)
    await session.flush()  # get doc.id

    db_chunks = [
        DocumentChunk(
            document_id=doc.id,
            chunk_index=c["chunk_index"],
            content=c["content"],
            token_count=c["token_count"],
            page_number=c.get("page_number"),
            source_section=c.get("source_section"),
            embedding=vectors[i],
        )
        for i, c in enumerate(chunks)
    ]
    session.add_all(db_chunks)
    await session.commit()

    logger.info("Ingested '%s' → %d chunks", file.filename, len(chunks))

    return IngestionResult(
        document_id=doc.id,
        filename=doc.filename,
        total_chunks=doc.total_chunks,
    )


@router.get("/documents", response_model=DocumentListOut, summary="List all documents")
async def list_documents(session: AsyncSession = Depends(get_session)):
    """Return all ingested documents ordered by creation date."""
    result = await session.execute(
        select(Document).order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return DocumentListOut(
        total=len(docs),
        documents=[DocumentOut.model_validate(d) for d in docs],
    )


@router.get("/documents/{document_id}", response_model=DocumentOut, summary="Get document details")
async def get_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Get details for a specific document."""
    doc = await session.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentOut.model_validate(doc)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
)
async def delete_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Delete a document and all associated chunks / embeddings."""
    doc = await session.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    await session.delete(doc)
    await session.commit()
    logger.info("Deleted document %s (%s)", document_id, doc.filename)
