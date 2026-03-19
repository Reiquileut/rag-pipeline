"""Generate embeddings for text chunks using OpenAI models."""

import logging
from typing import List

from langchain_openai import OpenAIEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)

_embeddings_model: OpenAIEmbeddings | None = None


def get_embeddings_model() -> OpenAIEmbeddings:
    """Singleton accessor for the embeddings model."""
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
    return _embeddings_model


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Returns a list of float vectors, one per input text.
    """
    model = get_embeddings_model()
    logger.info("Embedding %d texts with model %s", len(texts), settings.EMBEDDING_MODEL)
    vectors = await model.aembed_documents(texts)
    return vectors


async def embed_query(query: str) -> list[float]:
    """Generate a single embedding for a search query."""
    model = get_embeddings_model()
    return await model.aembed_query(query)
