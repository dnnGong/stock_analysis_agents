from __future__ import annotations

from openai import OpenAI

from .models import AgentResult


BASELINE_PROMPT = (
    "You are a financial assistant with no tool access. "
    "If you are unsure, be explicit about uncertainty and do not fabricate numbers."
)


def run_baseline(client: OpenAI, model: str, question: str) -> AgentResult:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BASELINE_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    answer = resp.choices[0].message.content or ""
    return AgentResult(agent_name="Baseline", answer=answer, tools_called=[])
