"""RAG chain: answer questions grounded in retrieved document chunks."""

import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings
from app.core.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """\
You are a precise research assistant. Answer the user's question based ONLY \
on the provided context chunks. Follow these rules strictly:

1. If the context contains the answer, provide a clear and concise response.
2. Cite your sources using [Source N] notation matching the chunk numbers below.
3. If the context does NOT contain enough information, say so explicitly — \
   do NOT hallucinate or guess.
4. Preserve technical accuracy and use the same terminology as the source docs.
"""

RAG_USER_PROMPT = """\
## Context

{context}

## Question

{question}

Provide a well-structured answer with [Source N] citations.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Build a numbered context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Source {i}] (file: {chunk.filename}"
        if chunk.page_number is not None:
            header += f", page {chunk.page_number}"
        header += f", similarity: {chunk.similarity_score:.2%})"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=2048,
    )


async def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
) -> dict[str, Any]:
    """
    Run the RAG chain: format context → prompt → LLM → parse.

    Returns:
        dict with keys: answer, model, usage
    """
    if not chunks:
        return {
            "answer": (
                "I couldn't find any relevant information in the uploaded documents "
                "to answer your question. Please try rephrasing or uploading "
                "additional documents."
            ),
            "model": settings.LLM_MODEL,
            "usage": None,
        }

    context = _format_context(chunks)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_PROMPT),
        ]
    )

    chain = prompt | _get_llm() | StrOutputParser()

    logger.info("Generating answer with %d context chunks", len(chunks))
    answer = await chain.ainvoke({"context": context, "question": question})

    return {
        "answer": answer,
        "model": settings.LLM_MODEL,
        "usage": None,  # Could be extended with callback handlers
    }
