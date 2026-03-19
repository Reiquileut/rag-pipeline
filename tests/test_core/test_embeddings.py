"""Tests for the embeddings module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import embeddings as embeddings_module
from app.core.embeddings import embed_query, embed_texts, get_embeddings_model


class TestGetEmbeddingsModel:
    def test_returns_model(self):
        embeddings_module._embeddings_model = None
        with patch("app.core.embeddings.OpenAIEmbeddings") as mock_cls:
            mock_cls.return_value = MagicMock()
            model = get_embeddings_model()
            assert model is not None
            mock_cls.assert_called_once()
        embeddings_module._embeddings_model = None

    def test_singleton_returns_same_instance(self):
        embeddings_module._embeddings_model = None
        with patch("app.core.embeddings.OpenAIEmbeddings") as mock_cls:
            sentinel = MagicMock()
            mock_cls.return_value = sentinel
            first = get_embeddings_model()
            second = get_embeddings_model()
            assert first is second
            mock_cls.assert_called_once()
        embeddings_module._embeddings_model = None


class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_embed_texts_returns_vectors(self):
        mock_model = MagicMock()
        mock_model.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        with patch("app.core.embeddings.get_embeddings_model", return_value=mock_model):
            result = await embed_texts(["hello", "world"])
            assert len(result) == 2
            assert result[0] == [0.1, 0.2]
            mock_model.aembed_documents.assert_called_once_with(["hello", "world"])


class TestEmbedQuery:
    @pytest.mark.asyncio
    async def test_embed_query_returns_vector(self):
        mock_model = MagicMock()
        mock_model.aembed_query = AsyncMock(return_value=[0.5, 0.6, 0.7])
        with patch("app.core.embeddings.get_embeddings_model", return_value=mock_model):
            result = await embed_query("search query")
            assert result == [0.5, 0.6, 0.7]
            mock_model.aembed_query.assert_called_once_with("search query")
