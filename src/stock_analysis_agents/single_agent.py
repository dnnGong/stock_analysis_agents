from __future__ import annotations

from openai import OpenAI

from .agent_runner import run_specialist_agent
from .models import AgentResult
from .schemas import ALL_SCHEMAS


SINGLE_AGENT_PROMPT = """
You are a financial analyst with tool access.
Rules:
1) Use tools for factual current values.
2) For multi-condition questions, verify each condition explicitly.
3) If data is missing or tool errors, say so and avoid guessing.
4) Include key numbers used in your conclusion.
"""


def run_single_agent(
    client: OpenAI,
    model: str,
    tool_functions: dict,
    question: str,
    verbose: bool = False,
) -> AgentResult:
    return run_specialist_agent(
        client=client,
        model=model,
        tool_functions=tool_functions,
        agent_name="Single Agent",
        system_prompt=SINGLE_AGENT_PROMPT,
        task=question,
        tool_schemas=ALL_SCHEMAS,
        max_iters=10,
        verbose=verbose,
    )
