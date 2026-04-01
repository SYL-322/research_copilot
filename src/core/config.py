"""Application settings loaded from environment (.env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for research_copilot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # --- OpenAI ---
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="Default chat model")
    openai_model_light: str | None = Field(
        default=None,
        description="Optional cheaper model for digest / batch tasks",
    )

    # --- Optional third-party APIs ---
    semantic_scholar_api_key: str | None = Field(
        default=None,
        description="Optional Semantic Scholar API key",
    )

    # --- Storage ---
    research_copilot_data_dir: Path | None = Field(
        default=None,
        validation_alias="RESEARCH_COPILOT_DATA_DIR",
        description="Root data directory; default: <project>/data",
    )
    database_filename: str = Field(
        default="research_copilot.db",
        validation_alias="DATABASE_FILENAME",
        description="SQLite filename under the data directory",
    )

    # --- HTTP ---
    http_timeout: float = Field(default=30.0, description="HTTP client timeout (seconds)")

    # --- Logging ---
    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Python logging level: DEBUG, INFO, WARNING, ERROR",
    )

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        upper = (v or "INFO").strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper

    def resolve_data_dir(self, project_root: Path) -> Path:
        """Return absolute data directory (does not create it)."""
        if self.research_copilot_data_dir is not None:
            return self.research_copilot_data_dir.expanduser().resolve()
        return (project_root / "data").resolve()

    def resolve_database_path(self, project_root: Path) -> Path:
        """SQLite file path under the configured data directory."""
        data = self.resolve_data_dir(project_root)
        name = self.database_filename.strip() or "research_copilot.db"
        if not name.endswith(".db"):
            name = f"{name}.db"
        return (data / name).resolve()

    def resolve_openai_model_main(self) -> str:
        """Primary / higher-quality chat model (``OPENAI_MODEL``)."""
        return (self.openai_model or "").strip() or "gpt-4o-mini"

    def resolve_openai_model_light(self) -> str:
        """Cheaper first-pass model; falls back to main if unset."""
        alt = (self.openai_model_light or "").strip()
        return alt if alt else self.resolve_openai_model_main()


@lru_cache
def load_settings() -> Settings:
    """Load and cache settings (call once per process)."""
    return Settings()
