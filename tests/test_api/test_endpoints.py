"""Tests for API endpoints."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()

        from app.db.database import get_session

        app.dependency_overrides[get_session] = lambda: mock_session
        response = await client.get("/health")
        app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_health_db_disconnected(self, client):
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=Exception("connection refused"))

        from app.db.database import get_session

        app.dependency_overrides[get_session] = lambda: mock_session
        response = await client.get("/health")
        app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["database"] == "disconnected"


class TestDocumentsEndpoint:
    @pytest.mark.asyncio
    async def test_list_documents_empty(self, client):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        from app.db.database import get_session

        app.dependency_overrides[get_session] = lambda: mock_session
        response = await client.get("/api/v1/documents")
        app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["documents"] == []

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, client):
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=None)

        from app.db.database import get_session

        app.dependency_overrides[get_session] = lambda: mock_session
        fake_id = uuid.uuid4()
        response = await client.get(f"/api/v1/documents/{fake_id}")
        app.dependency_overrides.clear()

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_upload_no_file(self, client):
        mock_session = AsyncMock()
        from app.db.database import get_session

        app.dependency_overrides[get_session] = lambda: mock_session
        response = await client.post("/api/v1/documents")
        app.dependency_overrides.clear()

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_upload_success(self, client):
        doc_id = uuid.uuid4()
        mock_session = AsyncMock()
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.add_all = MagicMock()

        mock_pages = [{"content": "Hello world", "page_number": 1, "source_section": None}]
        mock_chunks = [
            {"content": "Hello world", "chunk_index": 0, "token_count": 2, "page_number": 1}
        ]
        mock_vectors = [[0.1] * 1536]

        with (
            patch(
                "app.api.routes.documents.save_upload",
                new_callable=AsyncMock,
                return_value="/tmp/test.txt",
            ),
            patch(
                "app.api.routes.documents.extract_text",
                new_callable=AsyncMock,
                return_value=mock_pages,
            ),
            patch("app.api.routes.documents.chunk_pages", return_value=mock_chunks),
            patch(
                "app.api.routes.documents.embed_texts",
                new_callable=AsyncMock,
                return_value=mock_vectors,
            ),
            patch("app.api.routes.documents.Document") as mock_doc_cls,
        ):
            mock_doc = MagicMock()
            mock_doc.id = doc_id
            mock_doc.filename = "test.txt"
            mock_doc.total_chunks = 1
            mock_doc_cls.return_value = mock_doc

            from app.db.database import get_session

            app.dependency_overrides[get_session] = lambda: mock_session
            response = await client.post(
                "/api/v1/documents",
                files={"file": ("test.txt", b"Hello world", "text/plain")},
            )
            app.dependency_overrides.clear()

            assert response.status_code == 201
            data = response.json()
            assert "document_id" in data
            assert data["filename"] == "test.txt"
            assert data["total_chunks"] == 1

    @pytest.mark.asyncio
    async def test_upload_unsupported_type(self, client):
        from app.core.ingestion import UnsupportedFileTypeError

        mock_session = AsyncMock()

        with (
            patch(
                "app.api.routes.documents.save_upload",
                new_callable=AsyncMock,
                return_value="/tmp/test.xyz",
            ),
            patch(
                "app.api.routes.documents.extract_text",
                new_callable=AsyncMock,
                side_effect=UnsupportedFileTypeError("Unsupported"),
            ),
        ):
            from app.db.database import get_session

            app.dependency_overrides[get_session] = lambda: mock_session
            response = await client.post(
                "/api/v1/documents",
                files={"file": ("test.xyz", b"data", "application/octet-stream")},
            )
            app.dependency_overrides.clear()

            assert response.status_code == 415

    @pytest.mark.asyncio
    async def test_upload_empty_file(self, client):
        mock_session = AsyncMock()

        with (
            patch(
                "app.api.routes.documents.save_upload",
                new_callable=AsyncMock,
                return_value="/tmp/empty.txt",
            ),
            patch(
                "app.api.routes.documents.extract_text",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            from app.db.database import get_session

            app.dependency_overrides[get_session] = lambda: mock_session
            response = await client.post(
                "/api/v1/documents",
                files={"file": ("empty.txt", b"", "text/plain")},
            )
            app.dependency_overrides.clear()

            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_document_found(self, client):
        doc_id = uuid.uuid4()
        mock_doc = MagicMock()
        mock_doc.id = doc_id
        mock_doc.filename = "test.pdf"
        mock_doc.content_type = "application/pdf"
        mock_doc.total_chunks = 5
        mock_doc.file_size_bytes = 1024
        mock_doc.created_at = "2025-01-01T00:00:00"

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_doc)

        from app.db.database import get_session

        app.dependency_overrides[get_session] = lambda: mock_session
        response = await client.get(f"/api/v1/documents/{doc_id}")
        app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_delete_document_success(self, client):
        doc_id = uuid.uuid4()
        mock_doc = MagicMock()
        mock_doc.id = doc_id
        mock_doc.filename = "test.pdf"

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_doc)
        mock_session.delete = AsyncMock()
        mock_session.commit = AsyncMock()

        from app.db.database import get_session

        app.dependency_overrides[get_session] = lambda: mock_session
        response = await client.delete(f"/api/v1/documents/{doc_id}")
        app.dependency_overrides.clear()

        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, client):
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=None)

        from app.db.database import get_session

        app.dependency_overrides[get_session] = lambda: mock_session
        response = await client.delete(f"/api/v1/documents/{uuid.uuid4()}")
        app.dependency_overrides.clear()

        assert response.status_code == 404


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_empty_question_rejected(self, client):
        mock_session = AsyncMock()
        from app.db.database import get_session

        app.dependency_overrides[get_session] = lambda: mock_session
        response = await client.post(
            "/api/v1/chat",
            json={"question": ""},
        )
        app.dependency_overrides.clear()

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_valid_request(self, client):
        mock_chunks = []
        mock_answer = {
            "answer": "No relevant information found.",
            "model": "gpt-4o-mini",
            "usage": None,
        }

        with (
            patch("app.api.routes.chat.retrieve", new_callable=AsyncMock, return_value=mock_chunks),
            patch(
                "app.api.routes.chat.generate_answer",
                new_callable=AsyncMock,
                return_value=mock_answer,
            ),
        ):
            mock_session = AsyncMock()
            from app.db.database import get_session

            app.dependency_overrides[get_session] = lambda: mock_session
            response = await client.post(
                "/api/v1/chat",
                json={"question": "What is this about?"},
            )
            app.dependency_overrides.clear()

            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "citations" in data
            assert data["model"] == "gpt-4o-mini"
