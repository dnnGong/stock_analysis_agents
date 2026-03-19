from types import SimpleNamespace

import pandas as pd
import pytest

from stock_analysis_agents import providers


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_price_with_yfinance_computes_pct_change(monkeypatch):
    frame = pd.DataFrame({"Close": [100.0, 120.0]})

    monkeypatch.setattr(providers.yf, "download", lambda *args, **kwargs: frame)

    out = providers._price_with_yfinance(["AAPL"], period="1mo")

    assert out["AAPL"]["start_price"] == 100.0
    assert out["AAPL"]["end_price"] == 120.0
    assert out["AAPL"]["pct_change"] == 20.0
    assert out["AAPL"]["period"] == "1mo"


def test_price_with_yfinance_handles_empty_frame(monkeypatch):
    monkeypatch.setattr(providers.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    out = providers._price_with_yfinance(["AAPL"])

    assert out["AAPL"] == {"error": "No data — possibly delisted"}


def test_alphavantage_news_sentiment_extracts_article_fields(monkeypatch):
    payload = {
        "feed": [
            {
                "title": "Apple gains",
                "source": "Example",
                "overall_sentiment_label": "Bullish",
                "overall_sentiment_score": 0.8,
            }
        ]
    }
    monkeypatch.setattr(providers.requests, "get", lambda *args, **kwargs: FakeResponse(payload))
    provider = providers.AlphaVantageProvider(api_key="test")

    out = provider.get_news_sentiment("AAPL", limit=1)

    assert out == {
        "ticker": "AAPL",
        "articles": [
            {
                "title": "Apple gains",
                "source": "Example",
                "sentiment": "Bullish",
                "score": 0.8,
            }
        ],
    }


def test_alphavantage_overview_returns_error_when_name_missing(monkeypatch):
    monkeypatch.setattr(providers.requests, "get", lambda *args, **kwargs: FakeResponse({"Note": "limit"}))
    provider = providers.AlphaVantageProvider(api_key="test")

    out = provider.get_company_overview("AAPL")

    assert out == {"error": "No overview data for AAPL"}


def test_yahoo_finance_overview_handles_missing_name(monkeypatch):
    monkeypatch.setattr(providers.yf, "Ticker", lambda ticker: SimpleNamespace(info={}))
    provider = providers.YahooFinanceProvider()

    out = provider.get_company_overview("AAPL")

    assert out == {"error": "No overview data for AAPL"}


def test_hybrid_provider_falls_back_to_yahoo_for_overview():
    alpha = SimpleNamespace(get_company_overview=lambda ticker: {"error": "alpha failed"})
    yahoo = SimpleNamespace(get_company_overview=lambda ticker: {"ticker": ticker, "name": "Apple Inc."})
    provider = providers.HybridProvider(alpha=alpha, yahoo=yahoo)

    out = provider.get_company_overview("AAPL")

    assert out == {"ticker": "AAPL", "name": "Apple Inc."}


def test_make_data_provider_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        providers.make_data_provider("unknown", alphavantage_api_key="test")
