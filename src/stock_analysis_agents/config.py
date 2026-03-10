from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    alphavantage_api_key: str = ""
    data_provider: str = "alphavantage"
    model_small: str = "gpt-4o-mini"
    model_large: str = "gpt-4o"
    active_model: str = "gpt-4o-mini"
    db_path: Path = Path("stocks.db")



def load_settings() -> Settings:
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    data_provider = os.getenv("STOCK_AGENTS_DATA_PROVIDER", "alphavantage").strip().lower()
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY. Export it in your shell first.")
    if data_provider in {"alphavantage", "hybrid"} and not alphavantage_api_key:
        raise ValueError(
            "Missing ALPHAVANTAGE_API_KEY. It is required when STOCK_AGENTS_DATA_PROVIDER is "
            "'alphavantage' or 'hybrid'."
        )

    model_small = os.getenv("STOCK_AGENTS_MODEL_SMALL", "gpt-4o-mini")
    model_large = os.getenv("STOCK_AGENTS_MODEL_LARGE", "gpt-4o")
    active_model = os.getenv("STOCK_AGENTS_MODEL", model_small)
    db_path = Path(os.getenv("STOCK_AGENTS_DB_PATH", "stocks.db"))

    return Settings(
        openai_api_key=openai_api_key,
        alphavantage_api_key=alphavantage_api_key,
        data_provider=data_provider,
        model_small=model_small,
        model_large=model_large,
        active_model=active_model,
        db_path=db_path,
    )
