"""Tests for the document ingestion module."""

import os
import tempfile

import pytest

from app.core.ingestion import UnsupportedFileType, extract_text, save_upload


class TestSaveUpload:
    @pytest.mark.asyncio
    async def test_save_creates_file(self):
        content = b"Hello, this is a test file."
        path = await save_upload(content, "test.txt")
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read() == content
        os.unlink(path)


class TestExtractText:
    @pytest.mark.asyncio
    async def test_unsupported_type_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"data")
            f.flush()
            with pytest.raises(UnsupportedFileType):
                await extract_text(f.name, "application/octet-stream")
            os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_extract_txt(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("This is paragraph one.\n\nThis is paragraph two.")
            f.flush()
            pages = await extract_text(f.name, "text/plain")
            assert len(pages) > 0
            assert any("paragraph" in p["content"] for p in pages)
            os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_extract_empty_txt(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("")
            f.flush()
            pages = await extract_text(f.name, "text/plain")
            assert pages == []
            os.unlink(f.name)
