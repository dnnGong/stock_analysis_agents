from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .providers import MarketDataProvider


class FinanceTools:
    def __init__(self, provider: MarketDataProvider, db_path: Path):
        self.provider = provider
        self.db_path = db_path

    def get_price_performance(self, tickers: list[str], period: str = "1y") -> dict[str, Any]:
        return self.provider.get_price_performance(tickers=tickers, period=period)

    def get_market_status(self) -> dict[str, Any]:
        return self.provider.get_market_status()

    def get_top_gainers_losers(self) -> dict[str, Any]:
        return self.provider.get_top_gainers_losers()

    def get_news_sentiment(self, ticker: str, limit: int = 5) -> dict[str, Any]:
        return self.provider.get_news_sentiment(ticker=ticker, limit=limit)

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
        return self.provider.get_company_overview(ticker=ticker)

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
