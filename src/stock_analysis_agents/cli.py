from __future__ import annotations

import argparse
import json
import os

from .db_builder import create_local_database, get_distinct_sectors



def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Analysis Agents SDK CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("ask", help="Ask a single question")
    q.add_argument("question", type=str)
    q.add_argument("--arch", choices=["single", "multi"], default="multi")
    q.add_argument("--model", default=None, help="Override model")

    ev = sub.add_parser("eval", help="Run full benchmark evaluation")
    ev.add_argument("--output", default="results_sdk.xlsx")
    ev.add_argument("--model", default=None, help="Override model")

    db = sub.add_parser("build-db", help="Build stocks.db from CSV/XLSX input")
    db.add_argument("--input", required=True, help="Path to .csv or .xlsx with company data")
    db.add_argument("--db-path", default=None, help="Output sqlite db path (default: settings db path)")

    args = parser.parse_args()
    if args.cmd == "build-db":
        default_db = os.getenv("STOCK_AGENTS_DB_PATH", "stocks.db")
        db_path = args.db_path or default_db
        n = create_local_database(input_path=args.input, db_path=db_path)
        sectors = get_distinct_sectors(db_path=db_path)
        print(f"Built database: {db_path}")
        print(f"Rows loaded: {n}")
        print("Distinct sectors:")
        for s in sectors:
            print(f"- {s}")
        return

    from .benchmark import BENCHMARK_QUESTIONS
    from .config import load_settings
    from .evaluation import run_full_evaluation
    from .llm import make_client
    from .multi_agent import run_multi_agent
    from .single_agent import run_single_agent
    from .tools import FinanceTools, build_tool_function_map

    settings = load_settings()
    client = make_client(settings)
    model = args.model or settings.active_model

    tools = FinanceTools(settings.alphavantage_api_key, settings.db_path)
    tool_functions = build_tool_function_map(tools)

    if args.cmd == "ask":
        if args.arch == "single":
            out = run_single_agent(client, model, tool_functions, args.question, verbose=True)
            print(json.dumps(out.__dict__, ensure_ascii=False, indent=2, default=str))
        else:
            out = run_multi_agent(client, model, tool_functions, args.question, verbose=True)
            print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        return

    if args.cmd == "eval":
        path = run_full_evaluation(
            client=client,
            model=model,
            questions=BENCHMARK_QUESTIONS,
            tool_functions=tool_functions,
            output_xlsx=args.output,
        )
        print(f"Saved: {path}")
        return



if __name__ == "__main__":
    main()
