"""Text chunking strategies for splitting documents into embeddable pieces."""

import logging
from enum import Enum

import tiktoken
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from app.config import settings

logger = logging.getLogger(__name__)


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    TOKEN = "token"


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Return the token count for a piece of text."""
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def get_splitter(strategy: ChunkStrategy = ChunkStrategy.RECURSIVE):
    """Return a LangChain text splitter based on the chosen strategy."""
    if strategy == ChunkStrategy.TOKEN:
        return TokenTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_pages(
    pages: list[dict],
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
) -> list[dict]:
    """
    Split extracted pages into smaller chunks.

    Each returned dict contains:
      - content: str
      - chunk_index: int
      - page_number: int | None
      - source_section: str | None
      - token_count: int
    """
    splitter = get_splitter(strategy)
    chunks: list[dict] = []
    global_index = 0

    for page in pages:
        splits = splitter.split_text(page["content"])
        for split_text in splits:
            chunks.append(
                {
                    "content": split_text,
                    "chunk_index": global_index,
                    "page_number": page.get("page_number"),
                    "source_section": page.get("source_section"),
                    "token_count": count_tokens(split_text),
                }
            )
            global_index += 1

    logger.info("Chunked %d pages into %d chunks (strategy=%s)", len(pages), len(chunks), strategy.value)
    return chunks
