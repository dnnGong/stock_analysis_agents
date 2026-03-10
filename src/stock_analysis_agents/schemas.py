from __future__ import annotations


def _schema(name: str, desc: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


SCHEMA_TICKERS = _schema(
    "get_tickers_by_sector",
    "Return all stocks in a sector or industry from local database.",
    {"sector": {"type": "string", "description": "Sector or industry name"}},
    ["sector"],
)

SCHEMA_PRICE = _schema(
    "get_price_performance",
    "Get % price change for tickers over a period (1mo,3mo,6mo,ytd,1y).",
    {
        "tickers": {"type": "array", "items": {"type": "string"}},
        "period": {"type": "string", "default": "1y"},
    },
    ["tickers"],
)

SCHEMA_OVERVIEW = _schema(
    "get_company_overview",
    "Get fundamentals for one stock: P/E, EPS, market cap, 52-week high/low.",
    {"ticker": {"type": "string", "description": "Ticker symbol e.g. AAPL"}},
    ["ticker"],
)

SCHEMA_STATUS = _schema(
    "get_market_status",
    "Check if global stock exchanges are open or closed.",
    {},
    [],
)

SCHEMA_MOVERS = _schema(
    "get_top_gainers_losers",
    "Get today's top gainers, losers, and most active stocks.",
    {},
    [],
)

SCHEMA_NEWS = _schema(
    "get_news_sentiment",
    "Get latest news and sentiment for a stock.",
    {
        "ticker": {"type": "string"},
        "limit": {"type": "integer", "default": 5},
    },
    ["ticker"],
)

SCHEMA_SQL = _schema(
    "query_local_db",
    "Run a SQL SELECT query on stocks.db table stocks.",
    {"sql": {"type": "string", "description": "A valid SQL SELECT"}},
    ["sql"],
)

ALL_SCHEMAS = [
    SCHEMA_TICKERS,
    SCHEMA_PRICE,
    SCHEMA_OVERVIEW,
    SCHEMA_STATUS,
    SCHEMA_MOVERS,
    SCHEMA_NEWS,
    SCHEMA_SQL,
]

MARKET_TOOLS = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_STATUS, SCHEMA_MOVERS]
FUNDAMENTAL_TOOLS = [SCHEMA_OVERVIEW, SCHEMA_SQL, SCHEMA_TICKERS]
SENTIMENT_TOOLS = [SCHEMA_NEWS, SCHEMA_SQL]
