"""
All configuration loaded from environment variables.
NEVER hardcode credentials. Use .env file (not committed to git).
"""
from functools import lru_cache
from typing import Optional, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "META-AI"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Security
    SECRET_KEY: str = "change-me-in-production-at-least-32-chars"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    # LLM Providers
    META_AI_COOKIE: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.1-70b-versatile"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096
    LLM_TOP_P: float = 0.9

    # Database
    DATABASE_URL: str = "sqlite:///data/meta_ai.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Vector Database
    CHROMA_PERSIST_DIR: str = "data/vectordb"

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 30

    @field_validator("DEBUG", mode="before")
    @classmethod
    def _normalize_debug(cls, value: Any) -> bool:
        """Accept common env values like 'release' while keeping strict bool output."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "development"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
            raise ValueError(f"Unsupported DEBUG value: {value}")
        return bool(value)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
