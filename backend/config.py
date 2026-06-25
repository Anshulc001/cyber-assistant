"""Central configuration for the personal AI assistant.

Every environment-specific value lives here. Override any field with an
environment variable of the same name (e.g. `MODEL_NAME`, `API_PORT`) or a
`.env` file — nothing else in the codebase should hardcode these values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # --- App / server ---
    APP_NAME: str = "personal-ai-assistant"
    APP_VERSION: str = "0.1.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # CORS: the Cloudflare URL is dynamic, so allow all by default.
    # Tighten this once a stable hostname exists.
    CORS_ALLOW_ORIGINS: list[str] = ["*"]

    # --- LLM ---
    MODEL_NAME: str = "empero-ai/Qwythos-9B-Claude-Mythos-5-1M"
    MAX_NEW_TOKENS: int = 1024
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    CONTEXT_WINDOW: int = 8192

    # --- Embeddings / RAG ---
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384  # bge-small-en-v1.5 output dimension
    CHUNK_SIZE: int = 512  # characters per chunk
    CHUNK_OVERLAP: int = 64  # character overlap between chunks
    TOP_K: int = 4  # chunks retrieved per query

    # --- Google Drive storage roots ---
    # In Colab the Drive is mounted at /content/drive. Locally these paths
    # fall back to a ./data folder so the backend still runs off-GPU.
    DRIVE_ROOT: Path = Path("/content/drive/MyDrive/AI-Assistant")
    LOCAL_DATA_ROOT: Path = Path("./data")

    # Set True when running inside Colab with Drive mounted.
    USE_DRIVE: bool = False

    @property
    def storage_root(self) -> Path:
        return self.DRIVE_ROOT if self.USE_DRIVE else self.LOCAL_DATA_ROOT

    @property
    def models_dir(self) -> Path:
        return self.storage_root / "Models"

    @property
    def chat_history_dir(self) -> Path:
        return self.storage_root / "ChatHistory"

    @property
    def vector_db_dir(self) -> Path:
        return self.storage_root / "VectorDB"

    @property
    def uploads_dir(self) -> Path:
        return self.storage_root / "Uploads"

    @property
    def settings_dir(self) -> Path:
        return self.storage_root / "Settings"

    @property
    def database_dir(self) -> Path:
        return self.storage_root / "database"

    @property
    def database_path(self) -> Path:
        return self.database_dir / "assistant.db"

    def ensure_dirs(self) -> None:
        """Create every storage directory if it does not already exist."""
        for directory in (
            self.models_dir,
            self.chat_history_dir,
            self.vector_db_dir,
            self.uploads_dir,
            self.settings_dir,
            self.database_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the settings."""
    return Settings()


settings = get_settings()
