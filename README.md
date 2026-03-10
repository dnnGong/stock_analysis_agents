# stock-analysis-agents

A practical Python SDK for stock analysis with:
- baseline (no tools)
- single-agent (all tools)
- multi-agent (`orchestrator + specialists + critic`)
- evaluator and benchmark runner

This project is extracted from `mp3_assignment_chenlei1_dnngong2.ipynb` and packaged for reuse.

## Features
- Tooling for price performance, market status, top movers, company overview, news sentiment, and local SQL lookup.
- SDK APIs for `run_single_agent`, `run_multi_agent`, `run_evaluator`, and `run_full_evaluation`.
- CLI (`stock-agents`) for quick usage.
- Environment-variable-only secrets loading (no hardcoded keys).

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
```

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
### Ask one question (multi-agent default)
```bash
stock-agents ask "What is the P/E ratio of Apple (AAPL)?"
```

### Ask with single-agent
```bash
stock-agents ask "Compare the 1-year returns of AAPL, MSFT, GOOGL" --arch single
```

### Run full benchmark evaluation
```bash
stock-agents eval --model gpt-4o-mini --output results_sdk_mini.xlsx
stock-agents eval --model gpt-4o --output results_sdk_4o.xlsx
```

## Python SDK Usage
```python
from stock_analysis_agents import (
    load_settings,
    make_client,
    FinanceTools,
    build_tool_function_map,
    run_multi_agent,
)

settings = load_settings()
client = make_client(settings)
tools = FinanceTools(settings.alphavantage_api_key, settings.db_path)
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
  pyproject.toml
  README.md
  src/stock_analysis_agents/
    __init__.py
    config.py
    llm.py
    models.py
    tools.py
    schemas.py
    agent_runner.py
    baseline.py
    single_agent.py
    multi_agent.py
    evaluator.py
    benchmark.py
    evaluation.py
    cli.py
  tests/
```

## Notes
- All secrets are loaded from environment variables.
- `query_local_db` only allows `SELECT` statements for safety.
- `run_specialist_agent` limits tool calls per turn to reduce oversized API payload failures.
- Database builder supports both `.csv` and `.xlsx` inputs.
