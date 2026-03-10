# stock-analysis-agents

A practical Python SDK for stock analysis with:
- baseline (no tools)
- single-agent (all tools)
- multi-agent (`orchestrator + specialists + critic`)
- evaluator and benchmark runner

This project is extracted from `mp3_assignment_chenlei1_dnngong2.ipynb` and packaged for reuse.

## Features
- Tooling for price performance, market status, top movers, company overview, news sentiment, and local SQL lookup.
- Switchable data-source providers: `alphavantage`, `yahoo`, `hybrid`.
- SDK APIs for `run_single_agent`, `run_multi_agent`, `run_evaluator`, and `run_full_evaluation`.
- CLI (`stock-agents`) for quick usage.
- Environment-variable-only secrets loading (no hardcoded keys).

## Multi-Agent Architecture
```text
User Question
   |
   v
[Orchestrator]
   |- decides active specialists
   |- generates sub-tasks
   v
┌───────────────────────────────────────────────────────────┐
│ Specialists (tool-scoped)                                │
│   • Market Specialist      -> tickers/price/status/movers│
│   • Fundamental Specialist -> overview/sql/tickers       │
│   • Sentiment Specialist   -> news/sql                   │
└───────────────────────────────────────────────────────────┘
   |
   v
[Synthesis]
   |- merges specialist outputs into one draft answer
   v
[Critic]
   |- checks missing fields / contradictions / support
   |- outputs: confidence, issues, corrected final answer
   v
Final Answer (+ agent_results, elapsed_sec, architecture)
```

## Requirements
- Python 3.10+
- A local `stocks.db` file with table `stocks` (columns expected in the notebook assignment).
  - You can generate it with the built-in `build-db` command (see below).

## Installation
```bash
cd stock_analysis_agents
python -m pip install -e .
```

## Environment Variables
Export these in your shell before running:

```bash
export OPENAI_API_KEY="your_openai_key"
export ALPHAVANTAGE_API_KEY="your_alpha_vantage_key"

# Optional overrides
export STOCK_AGENTS_MODEL="gpt-4o-mini"
export STOCK_AGENTS_MODEL_SMALL="gpt-4o-mini"
export STOCK_AGENTS_MODEL_LARGE="gpt-4o"
export STOCK_AGENTS_DB_PATH="/absolute/path/to/stocks.db"
export STOCK_AGENTS_DATA_PROVIDER="hybrid"  # alphavantage | yahoo | hybrid
```

Provider notes:
- `alphavantage`: full endpoint coverage in this SDK (requires `ALPHAVANTAGE_API_KEY`).
- `yahoo`: good for price + overview; market-status/movers/news sentiment return `error` (unsupported).
- `hybrid`: Alpha Vantage for status/movers/news, Yahoo for price, overview fallback.

## Build `stocks.db` from your data file
If you have a companies dataset in `.csv` or `.xlsx`, build the local sqlite DB:

```bash
stock-agents build-db --input /path/to/sp500_companies.csv
```

or:

```bash
stock-agents build-db --input /path/to/companies.xlsx --db-path /path/to/stocks.db
```

Expected input columns (case-insensitive, common variants supported):
- `symbol`/`ticker`
- `shortname`/`company`
- `sector`
- `industry`
- `exchange`
- optional `marketcap` (used to derive `Large` / `Mid` / `Small`)

## CLI Usage
### Built-in help
```bash
stock-agents -h
stock-agents ask -h
stock-agents eval -h
stock-agents build-db -h
```

### Ask one question (multi-agent default)
```bash
stock-agents ask "What is the P/E ratio of Apple (AAPL)?"
```

### Ask with single-agent
```bash
stock-agents ask "Compare the 1-year returns of AAPL, MSFT, GOOGL" --arch single
```

### Ask with explicit provider
```bash
stock-agents ask "Are US markets open right now?" --provider alphavantage
stock-agents ask "Compare AAPL and MSFT 1-year return" --provider yahoo
```

### Run full benchmark evaluation
```bash
stock-agents eval --model gpt-4o-mini --output results_sdk_mini.xlsx
stock-agents eval --model gpt-4o --output results_sdk_4o.xlsx
stock-agents eval --model gpt-4o-mini --provider hybrid --output results_sdk_hybrid.xlsx
```

## Python SDK Usage
```python
from stock_analysis_agents import (
    load_settings,
    make_client,
    make_data_provider,
    FinanceTools,
    build_tool_function_map,
    run_multi_agent,
)

settings = load_settings()
client = make_client(settings)
provider = make_data_provider(settings.data_provider, settings.alphavantage_api_key)
tools = FinanceTools(provider=provider, db_path=settings.db_path)
func_map = build_tool_function_map(tools)

out = run_multi_agent(
    client=client,
    model=settings.active_model,
    tool_functions=func_map,
    question="For the top 3 semiconductor stocks by 1-year return, what are their P/E ratios?",
    verbose=False,
)
print(out["final_answer"])
```

## Project Structure
```text
stock_analysis_agents/
├── pyproject.toml
├── README.md
├── src/
│   └── stock_analysis_agents/
│       ├── __init__.py
│       ├── cli.py              # CLI entrypoint: stock-agents
│       ├── config.py           # env-based settings
│       ├── llm.py              # OpenAI client factory
│       ├── models.py           # shared dataclasses
│       ├── providers.py        # data-source abstraction layer
│       ├── tools.py            # tool wrappers + local DB queries
│       ├── schemas.py          # tool schemas for function calling
│       ├── agent_runner.py     # reusable tool-call loop
│       ├── baseline.py         # no-tool baseline agent
│       ├── single_agent.py     # single-agent architecture
│       ├── multi_agent.py      # orchestrator + specialists + critic
│       ├── evaluator.py        # LLM-as-judge scoring
│       ├── benchmark.py        # fixed benchmark question set
│       ├── evaluation.py       # batch runner + xlsx output
│       └── db_builder.py       # build stocks.db from csv/xlsx
└── tests/
    ├── test_imports.py
    └── test_db_builder.py
```

## Notes
- All secrets are loaded from environment variables.
- `query_local_db` only allows `SELECT` statements for safety.
- `run_specialist_agent` limits tool calls per turn to reduce oversized API payload failures.
- Database builder supports both `.csv` and `.xlsx` inputs.
