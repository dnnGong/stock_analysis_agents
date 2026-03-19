import sqlite3
from pathlib import Path

from stock_analysis_agents.tools import FinanceTools, build_tool_function_map


class DummyProvider:
    def get_price_performance(self, tickers, period="1y"):
        return {"tickers": tickers, "period": period}

    def get_market_status(self):
        return {"market": "open"}

    def get_top_gainers_losers(self):
        return {"gainers": ["AAPL"]}

    def get_news_sentiment(self, ticker, limit=5):
        return {"ticker": ticker, "limit": limit}

    def get_company_overview(self, ticker):
        return {"ticker": ticker, "pe_ratio": 25}


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE stocks (ticker TEXT, company TEXT, sector TEXT, industry TEXT, exchange TEXT)"
        )
        conn.executemany(
            "INSERT INTO stocks VALUES (?, ?, ?, ?, ?)",
            [
                ("AAPL", "Apple Inc.", "Technology", "Consumer Electronics", "NASDAQ"),
                ("NVDA", "NVIDIA Corp.", "Technology", "Semiconductors", "NASDAQ"),
                ("XOM", "Exxon Mobil", "Energy", "Oil & Gas", "NYSE"),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_query_local_db_rejects_non_select(tmp_path):
    tools = FinanceTools(DummyProvider(), tmp_path / "stocks.db")

    out = tools.query_local_db("DELETE FROM stocks")

    assert out == {"error": "Only SQL SELECT statements are allowed."}


def test_query_local_db_returns_rows(tmp_path):
    db_path = tmp_path / "stocks.db"
    _make_db(db_path)
    tools = FinanceTools(DummyProvider(), db_path)

    out = tools.query_local_db("SELECT ticker, sector FROM stocks ORDER BY ticker")

    assert out["columns"] == ["ticker", "sector"]
    assert [row["ticker"] for row in out["rows"]] == ["AAPL", "NVDA", "XOM"]


def test_get_tickers_by_sector_prefers_exact_sector_match(tmp_path):
    db_path = tmp_path / "stocks.db"
    _make_db(db_path)
    tools = FinanceTools(DummyProvider(), db_path)

    out = tools.get_tickers_by_sector("Technology")

    assert out["sector"] == "Technology"
    assert [row["ticker"] for row in out["stocks"]] == ["AAPL", "NVDA"]


def test_get_tickers_by_sector_falls_back_to_industry_like(tmp_path):
    db_path = tmp_path / "stocks.db"
    _make_db(db_path)
    tools = FinanceTools(DummyProvider(), db_path)

    out = tools.get_tickers_by_sector("Semi")

    assert [row["ticker"] for row in out["stocks"]] == ["NVDA"]


def test_build_tool_function_map_exposes_expected_tools(tmp_path):
    tools = FinanceTools(DummyProvider(), tmp_path / "stocks.db")

    tool_map = build_tool_function_map(tools)

    assert set(tool_map) == {
        "get_tickers_by_sector",
        "get_price_performance",
        "get_company_overview",
        "get_market_status",
        "get_top_gainers_losers",
        "get_news_sentiment",
        "query_local_db",
    }
