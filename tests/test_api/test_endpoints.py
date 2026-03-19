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
        with patch("app.api.routes.health.get_session") as mock_dep:
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock()
            mock_dep.return_value = mock_session

            # Override the dependency
            from app.db.database import get_session

            app.dependency_overrides[get_session] = lambda: mock_session
            response = await client.get("/health")
            app.dependency_overrides.clear()

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "version" in data


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
            patch("app.api.routes.chat.generate_answer", new_callable=AsyncMock, return_value=mock_answer),
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
