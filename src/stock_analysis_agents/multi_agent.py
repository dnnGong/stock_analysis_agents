from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI

from .agent_runner import run_specialist_agent
from .models import AgentResult
from .schemas import FUNDAMENTAL_TOOLS, MARKET_TOOLS, SENTIMENT_TOOLS
from .structured_logging import log_event


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

CRITIC_DRAFT_SCORE_PROMPT = """
You are evaluating one candidate final answer for a finance QA task.
Return STRICT JSON only:
{"confidence": 0.0, "issues": ["..."]}
Rules:
- confidence in [0,1]
- issues should list concrete deficits (missing fields, weak support, contradictions)
- output JSON only
"""

MINIMAL_REWRITE_PROMPT = """
You are a careful post-editor for finance QA answers.
Goal: minimally revise the draft answer.
Rules:
- Keep all supported numbers and claims unchanged where possible.
- Only patch missing required fields, obvious contradictions, or unsupported claims.
- Do NOT add generic disclaimers unless absolutely necessary.
- If no safe improvement is possible, return the original draft.
Return plain text only.
"""

DEFAULT_CRITIC_STRATEGY = "strict-rewrite"
VALID_CRITIC_STRATEGIES = {
    "strict-rewrite",
    "no-rewrite",
    "soft-gated",
    "dual-draft",
    "minimal-rewrite",
    "auto",
}


def _parse_json(text: str, fallback: dict) -> dict:
    try:
        return json.loads((text or "").strip())
    except Exception:
        return fallback


def _normalize_critic_strategy(strategy: str | None) -> str:
    raw = (strategy or DEFAULT_CRITIC_STRATEGY).strip().lower().replace("_", "-")
    aliases = {
        "strict": "strict-rewrite",
        "default": "strict-rewrite",
        "none": "no-rewrite",
        "no": "no-rewrite",
        "soft": "soft-gated",
        "soft-gate": "soft-gated",
        "dual": "dual-draft",
        "dual-draft-choice": "dual-draft",
        "minimal": "minimal-rewrite",
    }
    norm = aliases.get(raw, raw)
    if norm not in VALID_CRITIC_STRATEGIES:
        raise ValueError(
            "Unknown critic strategy. Use one of: "
            "strict-rewrite, no-rewrite, soft-gated, dual-draft, minimal-rewrite, auto."
        )
    return norm


def _infer_auto_critic_strategy(question: str) -> tuple[str, str]:
    q = (question or "").lower()

    domain_count = 0
    market_keys = ["price", "return", "month", "year", "52-week", "open", "market"]
    fund_keys = ["p/e", "pe ratio", "market cap", "fundamental", "eps", "large-cap"]
    sent_keys = ["sentiment", "news", "headline", "bullish", "bearish"]
    if any(k in q for k in market_keys):
        domain_count += 1
    if any(k in q for k in fund_keys):
        domain_count += 1
    if any(k in q for k in sent_keys):
        domain_count += 1

    hard_markers = [
        "top 3",
        "which",
        " but ",
        " and ",
        "for each",
        "closer to",
        "have grown more than",
        "return the",
        "compare",
    ]
    complexity_score = sum(1 for m in hard_markers if m in q)

    if complexity_score >= 4 or domain_count >= 3:
        return "no-rewrite", "auto:no-rewrite(high-complexity)"
    if complexity_score >= 2 or domain_count == 2:
        return "soft-gated", "auto:soft-gated(medium-complexity)"
    return "strict-rewrite", "auto:strict-rewrite(low-complexity)"


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


def _pack_specialist_outputs(
    specialist_results: list[AgentResult],
    include_confidence_issues: bool = False,
) -> str:
    if not include_confidence_issues:
        return "\n\n".join(
            f"[{r.agent_name}]\nTools: {r.tools_called}\nAnswer: {r.answer}" for r in specialist_results
        )

    return "\n\n".join(
        (
            f"[{r.agent_name}]\n"
            f"Tools: {r.tools_called}\n"
            f"Confidence: {r.confidence:.0%}\n"
            f"Issues: {', '.join(r.issues_found) if r.issues_found else 'none'}\n"
            f"Answer: {r.answer}"
        )
        for r in specialist_results
    )


def _synthesize_draft(
    client: OpenAI,
    model: str,
    question: str,
    packed_specialists: str,
    extra_instruction: str | None = None,
) -> str:
    system_prompt = SYNTHESIS_PROMPT
    if extra_instruction:
        system_prompt = f"{SYNTHESIS_PROMPT} {extra_instruction}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nSpecialist outputs:\n{packed_specialists}",
            },
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or "Unable to synthesize answer."


def _critic_review_draft(
    client: OpenAI,
    model: str,
    question: str,
    packed_specialists: str,
    draft_answer: str,
) -> tuple[float, list[str], str]:
    crit = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CRITIC_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Specialist outputs:\n{packed_specialists}\n\n"
                    f"Draft answer:\n{draft_answer}"
                ),
            },
        ],
        temperature=0,
    )
    cdata = _parse_json(
        crit.choices[0].message.content or "",
        {"confidence": 0.6, "issues": ["Critic output not valid JSON."], "final_answer": draft_answer},
    )
    try:
        conf = float(cdata.get("confidence", 0.6))
    except Exception:
        conf = 0.6
    conf = max(0.0, min(1.0, conf))
    issues = cdata.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    clean_issues = [str(x) for x in issues]
    final_answer = cdata.get("final_answer", draft_answer) or draft_answer
    return conf, clean_issues, final_answer


def _critic_score_candidate(
    client: OpenAI,
    model: str,
    question: str,
    candidate_label: str,
    candidate_answer: str,
) -> tuple[float, list[str]]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CRITIC_DRAFT_SCORE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Candidate: {candidate_label}\n"
                    f"Answer:\n{candidate_answer}"
                ),
            },
        ],
        temperature=0,
    )
    data = _parse_json(resp.choices[0].message.content or "", {"confidence": 0.5, "issues": ["Invalid JSON"]})
    try:
        conf = float(data.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    issues = data.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    return conf, [str(x) for x in issues]


def _minimal_rewrite(
    client: OpenAI,
    model: str,
    question: str,
    packed_specialists: str,
    draft_answer: str,
    critic_issues: list[str],
) -> str:
    issues_text = "\n".join(f"- {x}" for x in critic_issues) if critic_issues else "- none"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": MINIMAL_REWRITE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Specialist outputs:\n{packed_specialists}\n\n"
                    f"Critic issues:\n{issues_text}\n\n"
                    f"Draft answer:\n{draft_answer}"
                ),
            },
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or draft_answer


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
    critic_strategy: str = DEFAULT_CRITIC_STRATEGY,
) -> dict:
    start = time.time()
    log_event(
        "multi_agent.start",
        architecture="orchestrator",
        model=model,
        critic_strategy_requested=critic_strategy,
        question_preview=(question or "")[:180],
    )
    requested_strategy = _normalize_critic_strategy(critic_strategy)
    strategy = requested_strategy
    strategy_note = requested_strategy
    if requested_strategy == "auto":
        strategy, strategy_note = _infer_auto_critic_strategy(question)

    plan = _orchestrate(client, model, question)
    log_event("multi_agent.plan", architecture="orchestrator", agents=plan.get("agents", []))

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

    packed_plain = _pack_specialist_outputs(specialists, include_confidence_issues=False)
    draft_a = _synthesize_draft(client, model, question, packed_plain)
    conf_a, issues_a, rewrite_a = _critic_review_draft(client, model, question, packed_plain, draft_a)

    rewrite_applied = False
    gate_triggered: bool | None = None
    draft_choice = "A"
    critic_score_a: float | None = None
    critic_score_b: float | None = None

    if strategy == "strict-rewrite":
        final_answer = rewrite_a
        final_conf = conf_a
        final_issues = issues_a
        rewrite_applied = True
        draft_choice = "rewrite"
    elif strategy == "no-rewrite":
        final_answer = draft_a
        final_conf = conf_a
        final_issues = issues_a
        rewrite_applied = False
        draft_choice = "A"
    elif strategy == "soft-gated":
        gate_triggered = conf_a < 0.58 or len(issues_a) >= 3
        final_answer = rewrite_a if gate_triggered else draft_a
        final_conf = conf_a
        final_issues = issues_a
        rewrite_applied = bool(gate_triggered)
        draft_choice = "rewrite" if gate_triggered else "A"
    elif strategy == "minimal-rewrite":
        if issues_a:
            final_answer = _minimal_rewrite(
                client,
                model,
                question,
                packed_plain,
                draft_a,
                issues_a,
            )
            rewrite_applied = final_answer != draft_a
            draft_choice = "minimal-rewrite" if rewrite_applied else "A"
        else:
            final_answer = draft_a
            rewrite_applied = False
            draft_choice = "A"
        final_conf = conf_a
        final_issues = issues_a
    else:  # dual-draft
        packed_with_tags = _pack_specialist_outputs(specialists, include_confidence_issues=True)
        draft_b = _synthesize_draft(
            client,
            model,
            question,
            packed_with_tags,
            extra_instruction=(
                "Use confidence/issues tags only to improve precision; "
                "do not add extra disclaimers unless required by missing data."
            ),
        )
        conf_draft_a, issues_draft_a = _critic_score_candidate(client, model, question, "Draft A", draft_a)
        conf_draft_b, issues_draft_b = _critic_score_candidate(client, model, question, "Draft B", draft_b)
        score_a = conf_draft_a - 0.04 * len(issues_draft_a)
        score_b = conf_draft_b - 0.04 * len(issues_draft_b)
        critic_score_a = score_a
        critic_score_b = score_b

        if score_b > score_a:
            final_answer = draft_b
            final_conf = conf_draft_b
            final_issues = issues_draft_b
            draft_choice = "B"
        else:
            final_answer = draft_a
            final_conf = conf_draft_a
            final_issues = issues_draft_a
            draft_choice = "A"
        rewrite_applied = draft_choice == "B"

    critic_result = AgentResult(
        agent_name="Critic",
        answer=final_answer,
        tools_called=[],
        confidence=final_conf,
        issues_found=[str(x) for x in final_issues],
        reasoning=f"requested={requested_strategy}; effective={strategy}; note={strategy_note}; choice={draft_choice}",
    )

    for res in specialists:
        if critic_result.issues_found:
            res.confidence = min(res.confidence, max(0.4, critic_result.confidence - 0.1))

    diagnostics: dict[str, Any] = {
        "critic_strategy_requested": requested_strategy,
        "critic_strategy_effective": strategy,
        "strategy_note": strategy_note,
        "rewrite_applied": rewrite_applied,
        "gate_triggered": gate_triggered,
        "draft_choice": draft_choice,
        "critic_score_a": critic_score_a,
        "critic_score_b": critic_score_b,
        "critic_confidence": final_conf,
        "critic_issue_count": len(final_issues),
    }
    log_event(
        "multi_agent.critic_decision",
        architecture="orchestrator",
        critic_strategy_requested=requested_strategy,
        critic_strategy_effective=strategy,
        rewrite_applied=rewrite_applied,
        gate_triggered=gate_triggered,
        draft_choice=draft_choice,
        critic_score_a=critic_score_a,
        critic_score_b=critic_score_b,
        critic_confidence=final_conf,
        critic_issue_count=len(final_issues),
    )
    log_event(
        "multi_agent.end",
        architecture=f"orchestrator-specialists-critic[{strategy}]",
        elapsed_sec=round(time.time() - start, 3),
        specialist_count=len(specialists),
    )

    return {
        "final_answer": critic_result.answer,
        "agent_results": specialists + [critic_result],
        "elapsed_sec": time.time() - start,
        "architecture": f"orchestrator-specialists-critic[{strategy}]",
        "diagnostics": diagnostics,
    }


def run_multi_agent_pipeline(
    client: OpenAI,
    model: str,
    tool_functions: dict,
    question: str,
    verbose: bool = False,
) -> dict:
    start = time.time()
    log_event(
        "multi_agent.start",
        architecture="sequential-pipeline",
        model=model,
        question_preview=(question or "")[:180],
    )

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

    out = {
        "final_answer": aggregator.answer,
        "agent_results": specialists + [aggregator],
        "elapsed_sec": time.time() - start,
        "architecture": "sequential-pipeline",
        "diagnostics": {
            "critic_strategy_requested": "n/a",
            "critic_strategy_effective": "n/a",
            "rewrite_applied": False,
            "gate_triggered": None,
            "draft_choice": "n/a",
            "critic_score_a": None,
            "critic_score_b": None,
            "critic_confidence": None,
            "critic_issue_count": 0,
        },
    }
    log_event(
        "multi_agent.end",
        architecture=out["architecture"],
        elapsed_sec=round(out["elapsed_sec"], 3),
        specialist_count=len(specialists),
    )
    return out


def run_multi_agent_parallel(
    client: OpenAI,
    model: str,
    tool_functions: dict,
    question: str,
    verbose: bool = False,
) -> dict:
    start = time.time()
    log_event(
        "multi_agent.start",
        architecture="parallel-specialists-aggregator",
        model=model,
        question_preview=(question or "")[:180],
    )

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

    out = {
        "final_answer": aggregator.answer,
        "agent_results": specialists + [aggregator],
        "elapsed_sec": time.time() - start,
        "architecture": "parallel-specialists-aggregator",
        "diagnostics": {
            "critic_strategy_requested": "n/a",
            "critic_strategy_effective": "n/a",
            "rewrite_applied": False,
            "gate_triggered": None,
            "draft_choice": "n/a",
            "critic_score_a": None,
            "critic_score_b": None,
            "critic_confidence": None,
            "critic_issue_count": 0,
        },
    }
    log_event(
        "multi_agent.end",
        architecture=out["architecture"],
        elapsed_sec=round(out["elapsed_sec"], 3),
        specialist_count=len(specialists),
    )
    return out


def run_multi_agent(
    client: OpenAI,
    model: str,
    tool_functions: dict,
    question: str,
    verbose: bool = False,
    architecture: str = "orchestrator",
    critic_strategy: str = DEFAULT_CRITIC_STRATEGY,
) -> dict:
    arch = (architecture or "orchestrator").strip().lower()
    if arch in {"orchestrator", "orchestrator-specialists-critic", "default"}:
        return run_multi_agent_orchestrator(
            client,
            model,
            tool_functions,
            question,
            verbose=verbose,
            critic_strategy=critic_strategy,
        )
    if arch in {"pipeline", "sequential", "sequential-pipeline"}:
        return run_multi_agent_pipeline(client, model, tool_functions, question, verbose=verbose)
    if arch in {"parallel", "parallel-specialists-aggregator"}:
        return run_multi_agent_parallel(client, model, tool_functions, question, verbose=verbose)
    raise ValueError(
        "Unknown multi-agent architecture. Use one of: orchestrator, pipeline, parallel."
    )
