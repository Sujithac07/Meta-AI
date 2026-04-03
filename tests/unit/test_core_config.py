from meta_ai_core.config import get_settings


def test_get_settings_defaults(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)

    settings = get_settings()

    assert settings.openai_model == "gpt-4o"
    assert settings.api_port == 8000
