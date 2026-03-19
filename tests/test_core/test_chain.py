"""Tests for the RAG chain module."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.config import settings
from app.core.chain import _format_context, generate_answer
from app.core.retriever import RetrievedChunk


@pytest.fixture
def sample_chunks():
    return [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="doc1.pdf",
            chunk_index=0,
            content="Python is a programming language.",
            page_number=1,
            similarity_score=0.95,
        ),
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="doc2.pdf",
            chunk_index=1,
            content="FastAPI is a web framework.",
            page_number=None,
            similarity_score=0.88,
        ),
    ]


class TestFormatContext:
    def test_formats_chunks_with_page(self, sample_chunks):
        result = _format_context(sample_chunks)
        assert "[Source 1]" in result
        assert "[Source 2]" in result
        assert "doc1.pdf" in result
        assert "page 1" in result
        assert "Python is a programming language." in result

    def test_formats_chunk_without_page(self, sample_chunks):
        result = _format_context(sample_chunks)
        assert "doc2.pdf" in result
        assert "FastAPI is a web framework." in result

    def test_empty_chunks(self):
        result = _format_context([])
        assert result == ""

    def test_similarity_formatted_as_percentage(self, sample_chunks):
        result = _format_context(sample_chunks)
        assert "95.00%" in result


class TestGenerateAnswer:
    @pytest.mark.asyncio
    async def test_generate_with_chunks(self, sample_chunks):
        with patch("app.core.chain._get_llm") as mock_llm:
            mock_chain = AsyncMock()
            mock_chain.ainvoke = AsyncMock(return_value="The answer is 42.")
            mock_llm.return_value = mock_chain
            # Patch the chain pipeline
            with patch("app.core.chain.StrOutputParser") as mock_parser:
                mock_parser.return_value = AsyncMock()
                # We need to mock the pipe operator chain
                mock_prompt = AsyncMock()
                with patch("app.core.chain.ChatPromptTemplate") as mock_pt:
                    mock_pt.from_messages.return_value = mock_prompt
                    mock_prompt.__or__ = lambda self, other: mock_prompt
                    mock_prompt.ainvoke = AsyncMock(return_value="The answer is 42.")

                    result = await generate_answer("What is the meaning?", sample_chunks)

                    assert result["answer"] == "The answer is 42."
                    assert result["model"] == settings.LLM_MODEL
                    assert "usage" in result

    @pytest.mark.asyncio
    async def test_generate_without_chunks(self):
        with patch("app.core.chain._get_llm"):
            mock_prompt = AsyncMock()
            with patch("app.core.chain.ChatPromptTemplate") as mock_pt:
                mock_pt.from_messages.return_value = mock_prompt
                mock_prompt.__or__ = lambda self, other: mock_prompt
                mock_prompt.ainvoke = AsyncMock(return_value="I don't have context.")

                result = await generate_answer("What is this?", [])

                assert result["answer"] == "I don't have context."
                assert result["model"] == settings.LLM_MODEL
                assert result["usage"] is None
