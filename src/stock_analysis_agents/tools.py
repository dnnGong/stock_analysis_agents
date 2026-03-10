from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf


class FinanceTools:
    def __init__(self, alphavantage_api_key: str, db_path: Path):
        self.alphavantage_api_key = alphavantage_api_key
        self.db_path = db_path

    def get_price_performance(self, tickers: list[str], period: str = "1y") -> dict[str, Any]:
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

    def get_market_status(self) -> dict[str, Any]:
        return requests.get(
            "https://www.alphavantage.co/query"
            f"?function=MARKET_STATUS&apikey={self.alphavantage_api_key}",
            timeout=15,
        ).json()

    def get_top_gainers_losers(self) -> dict[str, Any]:
        return requests.get(
            "https://www.alphavantage.co/query"
            f"?function=TOP_GAINERS_LOSERS&apikey={self.alphavantage_api_key}",
            timeout=15,
        ).json()

    def get_news_sentiment(self, ticker: str, limit: int = 5) -> dict[str, Any]:
        data = requests.get(
            "https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT&tickers={ticker}&limit={limit}&apikey={self.alphavantage_api_key}",
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

    def query_local_db(self, sql: str) -> dict[str, Any]:
        # Keep this SDK safe for public use: only allow SELECT statements.
        if not sql.strip().lower().startswith("select"):
            return {"error": "Only SQL SELECT statements are allowed."}

        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql, conn)
            conn.close()
            return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
        except Exception as exc:
            return {"error": str(exc)}

    def get_company_overview(self, ticker: str) -> dict[str, Any]:
        data = requests.get(
            "https://www.alphavantage.co/query"
            f"?function=OVERVIEW&symbol={ticker}&apikey={self.alphavantage_api_key}",
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

    def get_tickers_by_sector(self, sector: str) -> dict[str, Any]:
        try:
            conn = sqlite3.connect(self.db_path)
            exact = pd.read_sql_query(
                "SELECT ticker, company, industry FROM stocks WHERE lower(sector) = lower(?)",
                conn,
                params=(sector,),
            )
            if exact.empty:
                like = pd.read_sql_query(
                    "SELECT ticker, company, industry FROM stocks WHERE lower(industry) LIKE lower(?)",
                    conn,
                    params=(f"%{sector}%",),
                )
                out = like
            else:
                out = exact
            conn.close()
            return {"sector": sector, "stocks": out.to_dict(orient="records")}
        except Exception as exc:
            return {"error": str(exc)}



def build_tool_function_map(tools: FinanceTools) -> dict[str, Any]:
    return {
        "get_tickers_by_sector": tools.get_tickers_by_sector,
        "get_price_performance": tools.get_price_performance,
        "get_company_overview": tools.get_company_overview,
        "get_market_status": tools.get_market_status,
        "get_top_gainers_losers": tools.get_top_gainers_losers,
        "get_news_sentiment": tools.get_news_sentiment,
        "query_local_db": tools.query_local_db,
    }
