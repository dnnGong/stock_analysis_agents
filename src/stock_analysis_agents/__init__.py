from .benchmark import BENCHMARK_QUESTIONS
from .config import Settings, load_settings
from .db_builder import create_local_database, get_distinct_sectors, load_companies_file

# Optional imports: these require the runtime deps like openai.
try:
    from .baseline import run_baseline
    from .evaluation import run_full_evaluation
    from .evaluator import run_evaluator
    from .llm import make_client
    from .multi_agent import run_multi_agent
    from .single_agent import run_single_agent
    from .tools import FinanceTools, build_tool_function_map
except Exception:
    run_baseline = None
    run_full_evaluation = None
    run_evaluator = None
    make_client = None
    run_multi_agent = None
    run_single_agent = None
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
    "run_evaluator",
    "run_full_evaluation",
    "make_client",
    "FinanceTools",
    "build_tool_function_map",
]
