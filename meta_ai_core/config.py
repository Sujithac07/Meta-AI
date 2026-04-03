from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    """Environment-driven settings for production-safe configuration."""

    app_env: str
    log_level: str
    openai_api_key: str
    openai_model: str
    groq_api_key: str
    meta_ai_cookie: str
    database_url: str
    redis_url: str
    api_host: str
    api_port: int


def _get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_settings() -> Settings:
    return Settings(
        app_env=_get_env("APP_ENV", "development"),
        log_level=_get_env("LOG_LEVEL", "INFO"),
        openai_api_key=_get_env("OPENAI_API_KEY"),
        openai_model=_get_env("OPENAI_MODEL", "gpt-4o"),
        groq_api_key=_get_env("GROQ_API_KEY"),
        meta_ai_cookie=_get_env("META_AI_COOKIE"),
        database_url=_get_env("DATABASE_URL", "sqlite:///mlflow.db"),
        redis_url=_get_env("REDIS_URL", "redis://localhost:6379/0"),
        api_host=_get_env("API_HOST", "127.0.0.1"),
        api_port=int(_get_env("API_PORT", "8000")),
    )
