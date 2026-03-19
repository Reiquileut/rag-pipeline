# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-19

### Added

- **Document Ingestion** — Upload and process PDF, DOCX, TXT, and Markdown files with configurable chunking strategies (recursive character or token-based).
- **Semantic Search** — Vector similarity search via pgvector (cosine distance) with configurable `top_k`, similarity threshold, and document-scoped filtering.
- **Answer Generation** — Grounded answers with `[Source N]` citation notation, hallucination guardrails, and fallback to direct LLM response when no relevant context is found.
- **REST API** — Fully async FastAPI application with Pydantic v2 validation, Swagger UI (`/docs`), ReDoc (`/redoc`), and health check endpoint.
- **Database Layer** — Async SQLAlchemy 2.0 + asyncpg with PostgreSQL 16 and pgvector extension for vector storage.
- **Docker Support** — Multi-stage Dockerfile (production + dev with hot-reload) and one-command deployment via `docker compose up`.
- **CI/CD Pipeline** — GitHub Actions workflow: lint (Ruff) → test (pytest) → Docker build.
- **Test Suite** — 97% coverage with pytest + pytest-asyncio + httpx.
- **Code Quality** — Ruff for linting and formatting enforcement.

[1.0.0]: https://github.com/Reiquileut/rag-pipeline/releases/tag/v1.0.0
