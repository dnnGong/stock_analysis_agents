import sqlite3

import pandas as pd

from stock_analysis_agents.db_builder import create_local_database, get_distinct_sectors



def test_create_local_database_from_csv(tmp_path):
    csv_path = tmp_path / "companies.csv"
    db_path = tmp_path / "stocks.db"

    df = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "shortname": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "exchange": "NASDAQ",
                "marketcap": 3_000_000_000_000,
            },
            {
                "symbol": "MSFT",
                "shortname": "Microsoft Corporation",
                "sector": "Technology",
                "industry": "Software - Infrastructure",
                "exchange": "NASDAQ",
                "marketcap": 2_000_000_000_000,
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    n = create_local_database(csv_path, db_path)
    assert n == 2

    sectors = get_distinct_sectors(db_path)
    assert "Technology" in sectors

    conn = sqlite3.connect(db_path)
    try:
        out = pd.read_sql_query("SELECT ticker, market_cap FROM stocks ORDER BY ticker", conn)
    finally:
        conn.close()

    assert list(out["ticker"]) == ["AAPL", "MSFT"]
    assert set(out["market_cap"]) == {"Large"}
