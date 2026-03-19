"""Health-check endpoint."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_session
from app.models.schemas import HealthOut

router = APIRouter()


@router.get("/health", response_model=HealthOut)
async def health_check(session: AsyncSession = Depends(get_session)):
    """Return service health including database connectivity."""
    db_status = "connected"
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    return HealthOut(
        status="ok",
        version="1.0.0",
        database=db_status,
    )
