# stock-analysis-agents

A practical Python SDK for stock analysis with:
- baseline (no tools)
- single-agent (all tools)
- multi-agent with 3 patterns:
  - `orchestrator + specialists + critic`
  - `sequential pipeline`
  - `parallel specialists + aggregator`
- critic strategies for orchestrator mode:
  - `strict-rewrite` (default)
  - `no-rewrite`
  - `soft-gated`
  - `dual-draft`
  - `minimal-rewrite`
  - `auto` (question-aware strategy selector)
- evaluator and benchmark runner

This project is extracted from `mp3_assignment_chenlei1_dnngong2.ipynb` and packaged for reuse.

## Features
- Tooling for price performance, market status, top movers, company overview, news sentiment, and local SQL lookup.
- Switchable data-source providers: `alphavantage`, `yahoo`, `hybrid`.
- SDK APIs for `run_single_agent`, `run_multi_agent`, `run_evaluator`, and `run_full_evaluation`.
- CLI (`stock-agents`) for quick usage.
- Environment-variable-only secrets loading (no hardcoded keys).

## Multi-Agent Architectures
### 1) Orchestrator + Specialists + Critic
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
   |- outputs: confidence, issues, corrected final answer (strategy-dependent)
   v
Final Answer (+ agent_results, elapsed_sec, architecture)
```

Critic strategy layer (orchestrator only):
```text
                        +------------------------------+
Draft A --------------->| strict-rewrite              |--> final = critic rewrite
                        +------------------------------+
                        +------------------------------+
Draft A + critic score->| no-rewrite                  |--> final = Draft A
                        +------------------------------+
                        +------------------------------+
Draft A + critic score->| soft-gated                  |--> low risk: keep Draft A
                        |                              |--> high risk: use critic rewrite
                        +------------------------------+
                        +------------------------------+
Draft A + issues ------>| minimal-rewrite             |--> patch only risky/missing parts
                        +------------------------------+
Draft A + Draft B ----->| dual-draft                  |--> critic scores both drafts
                        |                              |--> pick higher-scoring draft
                        +------------------------------+
                        +------------------------------+
Question text --------->| auto                        |--> choose strict/no/soft by complexity
                        +------------------------------+
```

### 2) Sequential Pipeline
```text
User Question
   |
   v
[Agent 1: Market]
   |
   v
[Agent 2: Fundamental]  (sees Agent 1 output)
   |
   v
[Agent 3: Sentiment]    (sees Agent 1 + 2 outputs)
   |
   v
[Aggregator]
   |
   v
Final Answer
```

### 3) Parallel Specialists + Aggregator
```text
User Question
   ├── [Market Specialist] ─┐
   ├── [Fundamental Spec.] ─┼──> [Aggregator] -> Final Answer
   └── [Sentiment Spec.]  ──┘
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
export STOCK_AGENTS_STRUCTURED_LOG="1"      # optional: 1 enables JSONL logs
export STOCK_AGENTS_LOG_PATH="./stock_agents_events.jsonl"  # optional
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
stock-agents eval-strategies -h
stock-agents build-db -h
```

### Ask one question (multi-agent default)
```bash
stock-agents ask "What is the P/E ratio of Apple (AAPL)?"
```
Default output is a human-readable summary. Use `--json` for raw structured output, and
use `--trace` to print detailed tool-call logs.

### Choose multi-agent pattern
```bash
stock-agents ask "Top 3 semiconductor stocks by 1-year return" --arch multi --multi-arch orchestrator
stock-agents ask "Top 3 semiconductor stocks by 1-year return" --arch multi --multi-arch pipeline
stock-agents ask "Top 3 semiconductor stocks by 1-year return" --arch multi --multi-arch parallel
```

### Choose critic strategy (orchestrator mode)
```bash
stock-agents ask "What is Apple's P/E ratio?" --arch multi --multi-arch orchestrator --critic-strategy strict-rewrite
stock-agents ask "What is Apple's P/E ratio?" --arch multi --multi-arch orchestrator --critic-strategy no-rewrite
stock-agents ask "What is Apple's P/E ratio?" --arch multi --multi-arch orchestrator --critic-strategy soft-gated
stock-agents ask "What is Apple's P/E ratio?" --arch multi --multi-arch orchestrator --critic-strategy dual-draft
stock-agents ask "What is Apple's P/E ratio?" --arch multi --multi-arch orchestrator --critic-strategy minimal-rewrite
stock-agents ask "What is Apple's P/E ratio?" --arch multi --multi-arch orchestrator --critic-strategy auto
```

`ask` output now includes a **Critic Diagnostics** section (strategy, rewrite applied, gate, draft choice, critic confidence/issues).

### Ask with single-agent
```bash
stock-agents ask "Compare the 1-year returns of AAPL, MSFT, GOOGL" --arch single
```

### Ask with explicit provider
```bash
stock-agents ask "Are US markets open right now?" --provider alphavantage
stock-agents ask "Compare AAPL and MSFT 1-year return" --provider yahoo
stock-agents ask "What is the P/E ratio of Apple (AAPL)?" --provider yahoo --json
```

### Run full benchmark evaluation
```bash
stock-agents eval --model gpt-4o-mini --multi-arch orchestrator --output results_sdk_mini_orch.xlsx
stock-agents eval --model gpt-4o-mini --multi-arch pipeline --output results_sdk_mini_pipeline.xlsx
stock-agents eval --model gpt-4o-mini --multi-arch parallel --output results_sdk_mini_parallel.xlsx
stock-agents eval --model gpt-4o --multi-arch orchestrator --critic-strategy soft-gated --output results_sdk_4o_orch_soft.xlsx
stock-agents eval --model gpt-4o --multi-arch orchestrator --critic-strategy dual-draft --output results_sdk_4o_orch_dual.xlsx
stock-agents eval --model gpt-4o --multi-arch orchestrator --critic-strategy no-rewrite --output results_sdk_4o_orch_norewrite.xlsx
stock-agents eval --model gpt-4o --multi-arch orchestrator --critic-strategy minimal-rewrite --output results_sdk_4o_orch_minrewrite.xlsx
stock-agents eval --model gpt-4o --multi-arch orchestrator --critic-strategy auto --output results_sdk_4o_orch_auto.xlsx
```

Each evaluation xlsx now includes:
- `Results` (per-question scores + MA diagnostics columns)
- `Summary` (Q3-style accuracy by architecture and difficulty)
- `Calibration` (confidence-vs-score calibration metrics)

### Compare all critic strategies in one command
```bash
stock-agents eval-strategies --model gpt-4o --output-prefix results_strategy_compare
stock-agents eval-strategies --model gpt-4o --strategies strict-rewrite,no-rewrite,soft-gated,dual-draft,minimal-rewrite,auto
```

This command writes one xlsx per strategy plus a consolidated CSV summary:
- `results_strategy_compare_<strategy>.xlsx`
- `results_strategy_compare_summary.csv`

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
    architecture="orchestrator",  # orchestrator | pipeline | parallel
    critic_strategy="auto",       # strict-rewrite | no-rewrite | soft-gated | dual-draft | minimal-rewrite | auto
)
print(out["final_answer"])
print(out["diagnostics"])
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
│       ├── multi_agent.py      # orchestrator / pipeline / parallel multi-agent patterns
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
- Structured logs (JSONL) are disabled by default. Enable with:
  - `STOCK_AGENTS_STRUCTURED_LOG=1`
  - optional `STOCK_AGENTS_LOG_PATH=/path/to/events.jsonl`
  - event types include: `agent.*`, `multi_agent.*`, `evaluation.*`
  - each line is a JSON object, suitable for pandas/duckdb/ELK ingestion
