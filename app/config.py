"""Centralised application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = "RAG Pipeline"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # --- Database (PostgreSQL + pgvector) ---
    DATABASE_URL: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag_pipeline"
    DATABASE_URL_SYNC: str = "postgresql://rag:rag@localhost:5432/rag_pipeline"

    # --- LLM / Embeddings ---
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # --- Chunking ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # --- Retrieval ---
    RETRIEVAL_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.0

    # --- CORS ---
    CORS_ORIGINS: list[str] = ["*"]

    # --- Upload ---
    MAX_UPLOAD_SIZE_MB: int = 50
    UPLOAD_DIR: str = "/tmp/rag_uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
