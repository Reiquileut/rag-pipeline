"""Tests for the text chunking module."""

import pytest

from app.core.chunking import ChunkStrategy, chunk_pages, count_tokens


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("Hello")
        long = count_tokens("Hello, this is a much longer sentence with many words.")
        assert long > short


class TestChunkPages:
    @pytest.fixture
    def sample_pages(self):
        return [
            {
                "content": "This is a test paragraph. " * 100,
                "page_number": 1,
                "source_section": "intro",
            },
            {
                "content": "Another section with different content. " * 80,
                "page_number": 2,
                "source_section": "body",
            },
        ]

    def test_chunks_are_created(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        assert len(chunks) > 0

    def test_chunk_structure(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        for chunk in chunks:
            assert "content" in chunk
            assert "chunk_index" in chunk
            assert "token_count" in chunk
            assert chunk["token_count"] > 0

    def test_chunk_indices_are_sequential(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_recursive_strategy(self, sample_pages):
        chunks = chunk_pages(sample_pages, strategy=ChunkStrategy.RECURSIVE)
        assert len(chunks) > 0

    def test_token_strategy(self, sample_pages):
        chunks = chunk_pages(sample_pages, strategy=ChunkStrategy.TOKEN)
        assert len(chunks) > 0

    def test_empty_pages(self):
        chunks = chunk_pages([])
        assert chunks == []

    def test_page_metadata_preserved(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        # At least the first chunk should have page 1 metadata
        assert any(c["page_number"] == 1 for c in chunks)
