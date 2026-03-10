from stock_analysis_agents import BENCHMARK_QUESTIONS


def test_benchmark_loaded():
    assert len(BENCHMARK_QUESTIONS) == 15
