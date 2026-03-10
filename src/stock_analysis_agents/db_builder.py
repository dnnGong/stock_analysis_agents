from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_OUTPUT_COLUMNS = [
    "ticker",
    "company",
    "sector",
    "industry",
    "market_cap",
    "exchange",
]



def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "") for c in df.columns]
    return df



def _resolve_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Supports common variants from CSV/XLSX exports.
    rename_map = {
        "symbol": "ticker",
        "ticker": "ticker",
        "shortname": "company",
        "company": "company",
        "sector": "sector",
        "industry": "industry",
        "exchange": "exchange",
        "marketcap": "market_cap_raw",
        "market_cap": "market_cap_raw",
        "marketcapraw": "market_cap_raw",
    }

    chosen = {}
    for col in df.columns:
        if col in rename_map:
            chosen[col] = rename_map[col]

    out = df.rename(columns=chosen)
    needed_min = {"ticker", "company", "sector", "industry", "exchange"}
    missing_min = [c for c in needed_min if c not in out.columns]
    if missing_min:
        raise ValueError(
            "Missing required input columns after normalization: "
            + ", ".join(sorted(missing_min))
        )
    return out



def _cap_bucket(value: object) -> str:
    try:
        v = float(value)
        if v >= 10_000_000_000:
            return "Large"
        if v >= 2_000_000_000:
            return "Mid"
        return "Small"
    except Exception:
        return "Unknown"



def load_companies_file(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    elif suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx")

    df = _normalize_columns(df)
    df = _resolve_input_columns(df)

    if "market_cap_raw" not in df.columns:
        df["market_cap_raw"] = None

    df["market_cap"] = df["market_cap_raw"].apply(_cap_bucket)

    out = (
        df.dropna(subset=["ticker", "company"])
        .drop_duplicates(subset=["ticker"])
        [REQUIRED_OUTPUT_COLUMNS]
    )
    return out



def create_local_database(input_path: str | Path, db_path: str | Path = "stocks.db") -> int:
    df = load_companies_file(input_path)
    db = Path(db_path)
    conn = sqlite3.connect(db)
    try:
        df.to_sql("stocks", conn, if_exists="replace", index=False)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ticker ON stocks(ticker)")
        conn.commit()
        n = int(pd.read_sql_query("SELECT COUNT(*) AS n FROM stocks", conn).iloc[0]["n"])
    finally:
        conn.close()
    return n



def get_distinct_sectors(db_path: str | Path = "stocks.db") -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT DISTINCT sector FROM stocks WHERE sector IS NOT NULL ORDER BY sector", conn
        )
    finally:
        conn.close()
    return [str(x) for x in df["sector"].tolist()]
