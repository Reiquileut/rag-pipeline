"""Chat endpoint — ask questions about your uploaded documents."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.chain import generate_answer
from app.core.retriever import retrieve
from app.db.database import get_session
from app.models.schemas import ChatRequest, ChatResponse, Citation

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Ask a question")
async def chat(
    body: ChatRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Ask a question about your uploaded documents.

    The pipeline:
    1. Embeds your question.
    2. Retrieves the most similar chunks via pgvector cosine similarity.
    3. Sends the chunks + question to the LLM.
    4. Returns the answer with source citations.
    """
    # Retrieve relevant chunks
    chunks = await retrieve(
        session,
        body.question,
        document_ids=body.document_ids,
        top_k=body.top_k,
    )

    # Generate grounded answer
    result = await generate_answer(body.question, chunks)

    # Build citation objects
    citations = [
        Citation(
            document_id=c.document_id,
            filename=c.filename,
            chunk_index=c.chunk_index,
            content_preview=c.content[:300],
            page_number=c.page_number,
            similarity_score=c.similarity_score,
        )
        for c in chunks
    ]

    return ChatResponse(
        answer=result["answer"],
        citations=citations,
        model=result["model"],
        usage=result.get("usage"),
    )
