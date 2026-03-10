from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import requests
import yfinance as yf


class MarketDataProvider(Protocol):
    name: str

    def get_price_performance(self, tickers: list[str], period: str = "1y") -> dict[str, Any]: ...
    def get_market_status(self) -> dict[str, Any]: ...
    def get_top_gainers_losers(self) -> dict[str, Any]: ...
    def get_news_sentiment(self, ticker: str, limit: int = 5) -> dict[str, Any]: ...
    def get_company_overview(self, ticker: str) -> dict[str, Any]: ...



def _price_with_yfinance(tickers: list[str], period: str = "1y") -> dict[str, Any]:
    results: dict[str, Any] = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                results[ticker] = {"error": "No data — possibly delisted"}
                continue
            start = float(data["Close"].iloc[0].item())
            end = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price": round(end, 2),
                "pct_change": round((end - start) / start * 100, 2),
                "period": period,
            }
        except Exception as exc:
            results[ticker] = {"error": str(exc)}
    return results


@dataclass
class AlphaVantageProvider:
    api_key: str
    name: str = "alphavantage"

    def get_price_performance(self, tickers: list[str], period: str = "1y") -> dict[str, Any]:
        # Alpha Vantage daily limits are tight; keep price from yfinance.
        return _price_with_yfinance(tickers, period)

    def get_market_status(self) -> dict[str, Any]:
        return requests.get(
            "https://www.alphavantage.co/query"
            f"?function=MARKET_STATUS&apikey={self.api_key}",
            timeout=15,
        ).json()

    def get_top_gainers_losers(self) -> dict[str, Any]:
        return requests.get(
            "https://www.alphavantage.co/query"
            f"?function=TOP_GAINERS_LOSERS&apikey={self.api_key}",
            timeout=15,
        ).json()

    def get_news_sentiment(self, ticker: str, limit: int = 5) -> dict[str, Any]:
        data = requests.get(
            "https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT&tickers={ticker}&limit={limit}&apikey={self.api_key}",
            timeout=15,
        ).json()
        return {
            "ticker": ticker,
            "articles": [
                {
                    "title": a.get("title"),
                    "source": a.get("source"),
                    "sentiment": a.get("overall_sentiment_label"),
                    "score": a.get("overall_sentiment_score"),
                }
                for a in data.get("feed", [])[:limit]
            ],
        }

    def get_company_overview(self, ticker: str) -> dict[str, Any]:
        data = requests.get(
            "https://www.alphavantage.co/query"
            f"?function=OVERVIEW&symbol={ticker}&apikey={self.api_key}",
            timeout=15,
        ).json()

        if "Name" not in data:
            return {"error": f"No overview data for {ticker}"}

        return {
            "ticker": ticker,
            "name": data.get("Name", ""),
            "sector": data.get("Sector", ""),
            "pe_ratio": data.get("PERatio", ""),
            "eps": data.get("EPS", ""),
            "market_cap": data.get("MarketCapitalization", ""),
            "52w_high": data.get("52WeekHigh", ""),
            "52w_low": data.get("52WeekLow", ""),
        }


class YahooFinanceProvider:
    name = "yahoo"

    def get_price_performance(self, tickers: list[str], period: str = "1y") -> dict[str, Any]:
        return _price_with_yfinance(tickers, period)

    def get_market_status(self) -> dict[str, Any]:
        return {"error": "market_status is not supported by yahoo provider"}

    def get_top_gainers_losers(self) -> dict[str, Any]:
        return {"error": "top_gainers_losers is not supported by yahoo provider"}

    def get_news_sentiment(self, ticker: str, limit: int = 5) -> dict[str, Any]:
        return {"error": "news_sentiment is not supported by yahoo provider"}

    def get_company_overview(self, ticker: str) -> dict[str, Any]:
        try:
            info = yf.Ticker(ticker).info
        except Exception as exc:
            return {"error": f"No overview data for {ticker}: {exc}"}

        name = info.get("longName") or info.get("shortName")
        if not name:
            return {"error": f"No overview data for {ticker}"}

        return {
            "ticker": ticker,
            "name": name,
            "sector": info.get("sector", ""),
            "pe_ratio": info.get("forwardPE") or info.get("trailingPE") or "",
            "eps": info.get("epsTrailingTwelveMonths") or info.get("trailingEps") or "",
            "market_cap": info.get("marketCap") or "",
            "52w_high": info.get("fiftyTwoWeekHigh") or "",
            "52w_low": info.get("fiftyTwoWeekLow") or "",
        }


@dataclass
class HybridProvider:
    alpha: AlphaVantageProvider
    yahoo: YahooFinanceProvider
    name: str = "hybrid"

    def get_price_performance(self, tickers: list[str], period: str = "1y") -> dict[str, Any]:
        return self.yahoo.get_price_performance(tickers, period)

    def get_market_status(self) -> dict[str, Any]:
        return self.alpha.get_market_status()

    def get_top_gainers_losers(self) -> dict[str, Any]:
        return self.alpha.get_top_gainers_losers()

    def get_news_sentiment(self, ticker: str, limit: int = 5) -> dict[str, Any]:
        return self.alpha.get_news_sentiment(ticker, limit)

    def get_company_overview(self, ticker: str) -> dict[str, Any]:
        out = self.alpha.get_company_overview(ticker)
        if isinstance(out, dict) and "error" in out:
            return self.yahoo.get_company_overview(ticker)
        return out



def make_data_provider(name: str, alphavantage_api_key: str | None = None) -> MarketDataProvider:
    provider = (name or "alphavantage").strip().lower()

    if provider == "yahoo":
        return YahooFinanceProvider()

    if provider == "alphavantage":
        if not alphavantage_api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY is required for alphavantage provider")
        return AlphaVantageProvider(api_key=alphavantage_api_key)

    if provider == "hybrid":
        if not alphavantage_api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY is required for hybrid provider")
        return HybridProvider(alpha=AlphaVantageProvider(alphavantage_api_key), yahoo=YahooFinanceProvider())

    raise ValueError("Unknown provider. Use one of: alphavantage, yahoo, hybrid")
