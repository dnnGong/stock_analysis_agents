from __future__ import annotations

import json
import time

from openai import OpenAI

from .agent_runner import run_specialist_agent
from .models import AgentResult
from .schemas import FUNDAMENTAL_TOOLS, MARKET_TOOLS, SENTIMENT_TOOLS


SPECIALIST_CONFIG = {
    "market": {
        "name": "Market Specialist",
        "tools": MARKET_TOOLS,
        "prompt": (
            "Handle market-status and price-performance tasks. "
            "Use tools; do not invent tickers or numbers. "
            "When ticker list is large, prioritize top 25 by market cap if available."
        ),
    },
    "fundamental": {
        "name": "Fundamental Specialist",
        "tools": FUNDAMENTAL_TOOLS,
        "prompt": (
            "Handle fundamentals and sector/industry filtering. "
            "Use overview/sql/ticker tools and keep output concise."
        ),
    },
    "sentiment": {
        "name": "Sentiment Specialist",
        "tools": SENTIMENT_TOOLS,
        "prompt": (
            "Handle news sentiment; summarize bullish/bearish/neutral evidence."
        ),
    },
}

ORCHESTRATOR_PROMPT = """
You route finance questions to specialists: market, fundamental, sentiment.
Return STRICT JSON only:
{
  "agents": ["market", "fundamental", "sentiment"],
  "subtasks": {
    "market": "...",
    "fundamental": "...",
    "sentiment": "..."
  }
}
Rules:
- Include only needed agents.
- If uncertain, include more agents.
- Subtasks should mention the exact computation needed.
"""

SYNTHESIS_PROMPT = (
    "Synthesize specialist outputs into one answer. "
    "Do not invent numbers. Mention missing data explicitly."
)

CRITIC_PROMPT = """
You are a strict QA critic for finance answers.
Given question, specialist outputs, and draft answer:
1) detect unsupported claims/contradictions/missing required fields
2) estimate confidence in [0,1]
3) return corrected final answer if needed
Return STRICT JSON only:
{"confidence": 0.0, "issues": ["..."], "final_answer": "..."}
"""



def _parse_json(text: str, fallback: dict) -> dict:
    try:
        return json.loads((text or "").strip())
    except Exception:
        return fallback



def _orchestrate(client: OpenAI, model: str, question: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    plan = _parse_json(raw, {"agents": ["market", "fundamental", "sentiment"], "subtasks": {}})

    agents = [a for a in plan.get("agents", []) if a in SPECIALIST_CONFIG]
    if not agents:
        agents = ["market", "fundamental", "sentiment"]
    subtasks = plan.get("subtasks", {}) if isinstance(plan.get("subtasks", {}), dict) else {}
    for a in agents:
        subtasks.setdefault(a, f"Answer from {a} perspective with tool evidence: {question}")
    return {"agents": agents, "subtasks": subtasks}



def run_multi_agent(
    client: OpenAI,
    model: str,
    tool_functions: dict,
    question: str,
    verbose: bool = False,
) -> dict:
    start = time.time()
    plan = _orchestrate(client, model, question)

    specialists: list[AgentResult] = []
    for key in plan["agents"]:
        cfg = SPECIALIST_CONFIG[key]
        result = run_specialist_agent(
            client=client,
            model=model,
            tool_functions=tool_functions,
            agent_name=cfg["name"],
            system_prompt=cfg["prompt"],
            task=plan["subtasks"].get(key, question),
            tool_schemas=cfg["tools"],
            max_iters=8,
            max_tool_calls_per_turn=25,
            verbose=verbose,
        )
        result.confidence = 0.75
        specialists.append(result)

    packed = "\n\n".join(
        f"[{r.agent_name}]\nTools: {r.tools_called}\nAnswer: {r.answer}" for r in specialists
    )

    synth = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {"role": "user", "content": f"Question:\n{question}\n\nSpecialist outputs:\n{packed}"},
        ],
        temperature=0,
    )
    draft = synth.choices[0].message.content or "Unable to synthesize answer."

    crit = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CRITIC_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Specialist outputs:\n{packed}\n\n"
                    f"Draft answer:\n{draft}"
                ),
            },
        ],
        temperature=0,
    )
    cdata = _parse_json(
        crit.choices[0].message.content or "",
        {"confidence": 0.6, "issues": ["Critic output not valid JSON."], "final_answer": draft},
    )

    try:
        conf = float(cdata.get("confidence", 0.6))
    except Exception:
        conf = 0.6
    conf = max(0.0, min(1.0, conf))
    issues = cdata.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    final_answer = cdata.get("final_answer", draft) or draft

    critic_result = AgentResult(
        agent_name="Critic",
        answer=final_answer,
        tools_called=[],
        confidence=conf,
        issues_found=[str(x) for x in issues],
    )

    for res in specialists:
        if critic_result.issues_found:
            res.confidence = min(res.confidence, max(0.4, critic_result.confidence - 0.1))

    return {
        "final_answer": critic_result.answer,
        "agent_results": specialists + [critic_result],
        "elapsed_sec": time.time() - start,
        "architecture": "orchestrator-specialists-critic",
    }
