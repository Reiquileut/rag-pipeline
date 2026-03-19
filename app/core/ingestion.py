"""Document ingestion: parse uploaded files into raw text."""

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

from app.config import settings

SUPPORTED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "text/markdown": "md",
}


class UnsupportedFileTypeError(Exception):
    """Raised when the uploaded file type is not supported."""


async def extract_text(file_path: str, content_type: str) -> list[dict]:
    """
    Extract text from a file and return a list of dicts with keys:
      - content: str
      - page_number: int | None
      - source_section: str | None

    Each entry roughly corresponds to a page or logical section.
    """
    file_ext = SUPPORTED_TYPES.get(content_type)
    if file_ext is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {content_type}. "
            f"Supported: {', '.join(SUPPORTED_TYPES.values())}"
        )

    loader_map = {
        "pdf": PyPDFLoader,
        "docx": UnstructuredWordDocumentLoader,
        "txt": TextLoader,
        "md": TextLoader,
    }

    loader_cls = loader_map[file_ext]
    loader = loader_cls(file_path)
    raw_docs = loader.load()

    pages = []
    for i, doc in enumerate(raw_docs):
        text = doc.page_content.strip()
        if not text:
            continue
        pages.append(
            {
                "content": text,
                "page_number": doc.metadata.get("page", i + 1),
                "source_section": doc.metadata.get("source", None),
            }
        )

    return pages


async def save_upload(file_bytes: bytes, filename: str) -> str:
    """Persist uploaded bytes to a temp directory and return the path."""
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / filename
    file_path.write_bytes(file_bytes)
    return str(file_path)
