"""Tests for the retriever module."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.retriever import RetrievedChunk, retrieve


class TestRetrievedChunk:
    def test_dataclass_creation(self):
        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="test.pdf",
            chunk_index=0,
            content="Some text",
            page_number=1,
            similarity_score=0.92,
        )
        assert chunk.filename == "test.pdf"
        assert chunk.similarity_score == 0.92
        assert chunk.page_number == 1

    def test_dataclass_none_page(self):
        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="test.txt",
            chunk_index=0,
            content="Text",
            page_number=None,
            similarity_score=0.5,
        )
        assert chunk.page_number is None


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_returns_chunks(self):
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        mock_row = MagicMock()
        mock_row.id = chunk_id
        mock_row.document_id = doc_id
        mock_row.filename = "test.pdf"
        mock_row.chunk_index = 0
        mock_row.content = "Some relevant content"
        mock_row.page_number = 1
        mock_row.similarity = 0.95

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [mock_row]
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.core.retriever.embed_query", new_callable=AsyncMock, return_value=[0.1] * 1536
        ):
            chunks = await retrieve(mock_session, "What is this?")

        assert len(chunks) == 1
        assert chunks[0].filename == "test.pdf"
        assert chunks[0].similarity_score == 0.95
        assert chunks[0].content == "Some relevant content"

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.core.retriever.embed_query", new_callable=AsyncMock, return_value=[0.1] * 1536
        ):
            chunks = await retrieve(mock_session, "No match query")

        assert chunks == []

    @pytest.mark.asyncio
    async def test_retrieve_with_document_ids(self):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        doc_id = uuid.uuid4()
        with patch(
            "app.core.retriever.embed_query", new_callable=AsyncMock, return_value=[0.1] * 1536
        ):
            chunks = await retrieve(mock_session, "query", document_ids=[doc_id])

        assert chunks == []
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_custom_top_k(self):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "app.core.retriever.embed_query", new_callable=AsyncMock, return_value=[0.1] * 1536
        ):
            await retrieve(mock_session, "query", top_k=3)

        mock_session.execute.assert_called_once()
