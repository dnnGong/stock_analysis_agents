from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from typing import Any

from .db_builder import create_local_database, get_distinct_sectors


def _line(title: str, value: str) -> str:
    return f"{title:<14}: {value}"


def _clip(text: str, limit: int = 500) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 3].rstrip() + "..."


def _serialize_obj(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, list):
        return [_serialize_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize_obj(v) for k, v in obj.items()}
    return obj


def _print_single_summary(question: str, model: str, provider: str, out: Any) -> None:
    print("\n=== Stock Agents Result ===")
    print(_line("Architecture", "single-agent"))
    print(_line("Model", model))
    print(_line("Provider", provider))
    print(_line("Question", _clip(question, 120)))
    print(_line("Tools Used", ", ".join(out.tools_called) if out.tools_called else "none"))
    print("\nFinal Answer:")
    print(_clip(out.answer, 1000) or "(empty)")


def _print_multi_summary(question: str, model: str, provider: str, out: dict[str, Any]) -> None:
    results = out.get("agent_results", [])
    print("\n=== Stock Agents Result ===")
    print(_line("Architecture", str(out.get("architecture", "multi-agent"))))
    print(_line("Model", model))
    print(_line("Provider", provider))
    print(_line("Question", _clip(question, 120)))
    print(_line("Elapsed Sec", f"{out.get('elapsed_sec', 0):.2f}"))
    print(_line("Agents", ", ".join(r.agent_name for r in results) if results else "none"))

    print("\nFinal Answer:")
    print(_clip(str(out.get("final_answer", "")), 1000) or "(empty)")

    if results:
        print("\nAgent Breakdown:")
        for idx, r in enumerate(results, start=1):
            tools = ", ".join(r.tools_called) if r.tools_called else "none"
            issues = "; ".join(r.issues_found) if r.issues_found else "none"
            print(f"{idx}. {r.agent_name}")
            print(f"   tools      : {tools}")
            print(f"   confidence : {r.confidence:.0%}")
            print(f"   issues     : {issues}")


def _print_eval_summary(path: str) -> None:
    import pandas as pd

    try:
        df = pd.read_excel(path, sheet_name="Results")
    except Exception:
        print(f"Saved: {path}")
        return

    def _avg(col: str) -> float:
        if col not in df.columns:
            return float("nan")
        valid = df[df[col] >= 0][col]
        if len(valid) == 0:
            return float("nan")
        return float(valid.mean())

    print("\n=== Evaluation Summary ===")
    print(_line("Output", path))
    print(_line("Questions", str(len(df))))
    print(_line("Baseline Avg", f"{_avg('bl_score'):.2f}/3"))
    print(_line("Single Avg", f"{_avg('sa_score'):.2f}/3"))
    print(_line("Multi Avg", f"{_avg('ma_score'):.2f}/3"))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="stock-agents",
        description="Stock Analysis Agents SDK CLI",
        epilog=(
            "Examples:\n"
            "  stock-agents build-db --input ./sp500_companies.csv\n"
            "  stock-agents ask \"What is the P/E ratio of Apple (AAPL)?\"\n"
            "  stock-agents ask \"Compare AAPL and MSFT 1-year return\" --arch single --provider yahoo\n"
            "  stock-agents eval --model gpt-4o-mini --provider hybrid --output results_sdk.xlsx\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    q = sub.add_parser(
        "ask",
        help="Ask one question with single or multi agent",
        description="Run one question through the selected architecture.",
    )
    q.add_argument("question", type=str)
    q.add_argument(
        "--arch",
        choices=["single", "multi"],
        default="multi",
        help="Agent architecture to use (default: multi)",
    )
    q.add_argument(
        "--model",
        default=None,
        help="Override model name, e.g. gpt-4o-mini or gpt-4o",
    )
    q.add_argument(
        "--provider",
        choices=["alphavantage", "yahoo", "hybrid"],
        default=None,
        help=(
            "Data-source provider override. If omitted, uses STOCK_AGENTS_DATA_PROVIDER.\n"
            "alphavantage: full coverage; yahoo: price+overview only; hybrid: alpha+yahoo fallback."
        ),
    )
    q.add_argument(
        "--trace",
        action="store_true",
        help="Print per-tool call trace (verbose agent loop logs).",
    )
    q.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON output instead of human-readable summary.",
    )

    ev = sub.add_parser(
        "eval",
        help="Run full benchmark evaluation",
        description="Run baseline/single/multi on all benchmark questions and write an xlsx.",
    )
    ev.add_argument(
        "--output",
        default="results_sdk.xlsx",
        help="Output .xlsx file path (default: results_sdk.xlsx)",
    )
    ev.add_argument(
        "--model",
        default=None,
        help="Override model name for this run",
    )
    ev.add_argument(
        "--provider",
        choices=["alphavantage", "yahoo", "hybrid"],
        default=None,
        help="Data-source provider override for this run",
    )

    db = sub.add_parser(
        "build-db",
        help="Build stocks.db from CSV/XLSX input",
        description=(
            "Create local sqlite database from company universe file.\n"
            "Supported input types: .csv, .xlsx"
        ),
    )
    db.add_argument(
        "--input",
        required=True,
        help="Path to source file (.csv or .xlsx)",
    )
    db.add_argument(
        "--db-path",
        default=None,
        help="Output sqlite db path (default: STOCK_AGENTS_DB_PATH or ./stocks.db)",
    )

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
    from .providers import make_data_provider
    from .single_agent import run_single_agent
    from .tools import FinanceTools, build_tool_function_map

    settings = load_settings()
    client = make_client(settings)
    model = args.model or settings.active_model

    provider_name = getattr(args, "provider", None) or settings.data_provider
    provider = make_data_provider(provider_name, settings.alphavantage_api_key)
    tools = FinanceTools(provider=provider, db_path=settings.db_path)
    tool_functions = build_tool_function_map(tools)

    if args.cmd == "ask":
        if args.arch == "single":
            out = run_single_agent(client, model, tool_functions, args.question, verbose=args.trace)
            if args.json:
                print(json.dumps(_serialize_obj(out), ensure_ascii=False, indent=2, default=str))
            else:
                _print_single_summary(args.question, model, provider_name, out)
        else:
            out = run_multi_agent(client, model, tool_functions, args.question, verbose=args.trace)
            if args.json:
                print(json.dumps(_serialize_obj(out), ensure_ascii=False, indent=2, default=str))
            else:
                _print_multi_summary(args.question, model, provider_name, out)
        return

    if args.cmd == "eval":
        path = run_full_evaluation(
            client=client,
            model=model,
            questions=BENCHMARK_QUESTIONS,
            tool_functions=tool_functions,
            output_xlsx=args.output,
        )
        _print_eval_summary(path)
        return


if __name__ == "__main__":
    main()
