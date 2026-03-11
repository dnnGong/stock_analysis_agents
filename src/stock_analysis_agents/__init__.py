from .benchmark import BENCHMARK_QUESTIONS
from .config import Settings, load_settings
from .db_builder import create_local_database, get_distinct_sectors, load_companies_file
from .structured_logging import is_enabled as structured_log_enabled, log_event

# Optional imports: these require the runtime deps like openai.
try:
    from .baseline import run_baseline
    from .evaluation import run_full_evaluation
    from .evaluator import run_evaluator
    from .llm import make_client
    from .multi_agent import (
        run_multi_agent,
        run_multi_agent_orchestrator,
        run_multi_agent_parallel,
        run_multi_agent_pipeline,
    )
    from .single_agent import run_single_agent
    from .providers import (
        AlphaVantageProvider,
        HybridProvider,
        YahooFinanceProvider,
        make_data_provider,
    )
    from .tools import FinanceTools, build_tool_function_map
except Exception:
    run_baseline = None
    run_full_evaluation = None
    run_evaluator = None
    make_client = None
    run_multi_agent = None
    run_single_agent = None
    run_multi_agent_orchestrator = None
    run_multi_agent_pipeline = None
    run_multi_agent_parallel = None
    AlphaVantageProvider = None
    YahooFinanceProvider = None
    HybridProvider = None
    make_data_provider = None
    FinanceTools = None
    build_tool_function_map = None

__all__ = [
    "Settings",
    "load_settings",
    "load_companies_file",
    "create_local_database",
    "get_distinct_sectors",
    "BENCHMARK_QUESTIONS",
    "run_baseline",
    "run_single_agent",
    "run_multi_agent",
    "run_multi_agent_orchestrator",
    "run_multi_agent_pipeline",
    "run_multi_agent_parallel",
    "run_evaluator",
    "run_full_evaluation",
    "make_client",
    "make_data_provider",
    "AlphaVantageProvider",
    "YahooFinanceProvider",
    "HybridProvider",
    "FinanceTools",
    "build_tool_function_map",
    "log_event",
    "structured_log_enabled",
]
