from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            "Use overview/sql/ticker tools and keep output concise. "
            "If get_company_overview returns pe_ratio, use that directly."
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



def _run_one_specialist(
    client: OpenAI,
    model: str,
    tool_functions: dict,
    specialist_key: str,
    task: str,
    verbose: bool,
) -> AgentResult:
    cfg = SPECIALIST_CONFIG[specialist_key]
    result = run_specialist_agent(
        client=client,
        model=model,
        tool_functions=tool_functions,
        agent_name=cfg["name"],
        system_prompt=cfg["prompt"],
        task=task,
        tool_schemas=cfg["tools"],
        max_iters=8,
        max_tool_calls_per_turn=25,
        verbose=verbose,
    )
    result.confidence = 0.75
    return result



def _aggregate_answers(
    client: OpenAI,
    model: str,
    question: str,
    specialist_results: list[AgentResult],
    architecture_name: str,
) -> AgentResult:
    packed = "\n\n".join(
        f"[{r.agent_name}]\nTools: {r.tools_called}\nAnswer: {r.answer}" for r in specialist_results
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Architecture: {architecture_name}\n"
                    f"Question:\n{question}\n\n"
                    f"Specialist outputs:\n{packed}"
                ),
            },
        ],
        temperature=0,
    )
    return AgentResult(
        agent_name="Aggregator",
        answer=resp.choices[0].message.content or "Unable to synthesize answer.",
        tools_called=[],
        confidence=0.7,
        issues_found=[],
        reasoning="Merged specialist outputs into a single answer.",
    )



def run_multi_agent_orchestrator(
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
        result = _run_one_specialist(
            client=client,
            model=model,
            tool_functions=tool_functions,
            specialist_key=key,
            task=plan["subtasks"].get(key, question),
            verbose=verbose,
        )
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



def run_multi_agent_pipeline(
    client: OpenAI,
    model: str,
    tool_functions: dict,
    question: str,
    verbose: bool = False,
) -> dict:
    start = time.time()

    stage1_task = (
        "Step 1 (market discovery): extract relevant tickers/price constraints for this question.\n"
        f"Question: {question}"
    )
    s1 = _run_one_specialist(client, model, tool_functions, "market", stage1_task, verbose)

    stage2_task = (
        "Step 2 (fundamentals refinement): use Stage 1 output as context and add fundamentals.\n"
        f"Question: {question}\n\nStage 1 output:\n{s1.answer}"
    )
    s2 = _run_one_specialist(client, model, tool_functions, "fundamental", stage2_task, verbose)

    stage3_task = (
        "Step 3 (sentiment enrichment): use Stage 1 + Stage 2 outputs and add sentiment context.\n"
        f"Question: {question}\n\nStage 1 output:\n{s1.answer}\n\nStage 2 output:\n{s2.answer}"
    )
    s3 = _run_one_specialist(client, model, tool_functions, "sentiment", stage3_task, verbose)

    specialists = [s1, s2, s3]
    aggregator = _aggregate_answers(
        client=client,
        model=model,
        question=question,
        specialist_results=specialists,
        architecture_name="sequential-pipeline",
    )

    return {
        "final_answer": aggregator.answer,
        "agent_results": specialists + [aggregator],
        "elapsed_sec": time.time() - start,
        "architecture": "sequential-pipeline",
    }



def run_multi_agent_parallel(
    client: OpenAI,
    model: str,
    tool_functions: dict,
    question: str,
    verbose: bool = False,
) -> dict:
    start = time.time()

    tasks = {
        "market": f"Price/market view for question: {question}",
        "fundamental": f"Fundamental view for question: {question}",
        "sentiment": f"Sentiment/news view for question: {question}",
    }

    specialists: list[AgentResult] = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(
                _run_one_specialist,
                client,
                model,
                tool_functions,
                key,
                tasks[key],
                verbose,
            ): key
            for key in ("market", "fundamental", "sentiment")
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                specialists.append(fut.result())
            except Exception as exc:
                specialists.append(
                    AgentResult(
                        agent_name=SPECIALIST_CONFIG[key]["name"],
                        answer=f"Specialist failed: {exc}",
                        tools_called=[],
                        confidence=0.0,
                        issues_found=[str(exc)],
                    )
                )

    # Keep deterministic ordering in output.
    name_order = {
        SPECIALIST_CONFIG["market"]["name"]: 0,
        SPECIALIST_CONFIG["fundamental"]["name"]: 1,
        SPECIALIST_CONFIG["sentiment"]["name"]: 2,
    }
    specialists.sort(key=lambda r: name_order.get(r.agent_name, 99))

    aggregator = _aggregate_answers(
        client=client,
        model=model,
        question=question,
        specialist_results=specialists,
        architecture_name="parallel-specialists-aggregator",
    )

    return {
        "final_answer": aggregator.answer,
        "agent_results": specialists + [aggregator],
        "elapsed_sec": time.time() - start,
        "architecture": "parallel-specialists-aggregator",
    }



def run_multi_agent(
    client: OpenAI,
    model: str,
    tool_functions: dict,
    question: str,
    verbose: bool = False,
    architecture: str = "orchestrator",
) -> dict:
    arch = (architecture or "orchestrator").strip().lower()
    if arch in {"orchestrator", "orchestrator-specialists-critic", "default"}:
        return run_multi_agent_orchestrator(client, model, tool_functions, question, verbose=verbose)
    if arch in {"pipeline", "sequential", "sequential-pipeline"}:
        return run_multi_agent_pipeline(client, model, tool_functions, question, verbose=verbose)
    if arch in {"parallel", "parallel-specialists-aggregator"}:
        return run_multi_agent_parallel(client, model, tool_functions, question, verbose=verbose)
    raise ValueError(
        "Unknown multi-agent architecture. Use one of: orchestrator, pipeline, parallel."
    )
