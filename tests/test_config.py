import pytest

from stock_analysis_agents.config import load_settings


def test_load_settings_allows_yahoo_without_alpha_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.setenv("STOCK_AGENTS_DATA_PROVIDER", "yahoo")

    settings = load_settings()

    assert settings.openai_api_key == "test-openai"
    assert settings.data_provider == "yahoo"
    assert settings.alphavantage_api_key == ""


def test_load_settings_requires_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.setenv("STOCK_AGENTS_DATA_PROVIDER", "yahoo")

    with pytest.raises(ValueError, match="Missing OPENAI_API_KEY"):
        load_settings()


def test_load_settings_requires_alpha_key_for_hybrid(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.setenv("STOCK_AGENTS_DATA_PROVIDER", "hybrid")

    with pytest.raises(ValueError, match="Missing ALPHAVANTAGE_API_KEY"):
        load_settings()
