from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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

DEFAULT_SOFT_GATE_THRESHOLDS = {
    "global": {"conf": 0.58, "issues": 3},
    "easy": {"conf": 0.55, "issues": 4},
    "medium": {"conf": 0.58, "issues": 3},
    "hard": {"conf": 0.62, "issues": 2},
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


def _normalize_data_mode(mode: str | None) -> str:
    raw = (mode or "off").strip().lower()
    aliases = {
        "none": "off",
        "disabled": "off",
        "global": "global",
        "stratified": "stratified",
    }
    norm = aliases.get(raw, raw)
    if norm not in {"off", "global", "stratified"}:
        raise ValueError("soft_gate_data_mode must be one of: off, global, stratified.")
    return norm


def _normalize_difficulty(difficulty: str | None, question: str | None = None) -> str:
    d = (difficulty or "").strip().lower()
    if d in {"easy", "medium", "hard"}:
        return d
    q = (question or "").lower()
    if "top 3" in q or "which" in q and (" and " in q or " but " in q):
        return "hard"
    if "compare" in q or "sentiment" in q or "6-month" in q:
        return "medium"
    return "easy"


def _coerce_thresholds(raw: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    out = {
        "global": dict(DEFAULT_SOFT_GATE_THRESHOLDS["global"]),
        "easy": dict(DEFAULT_SOFT_GATE_THRESHOLDS["easy"]),
        "medium": dict(DEFAULT_SOFT_GATE_THRESHOLDS["medium"]),
        "hard": dict(DEFAULT_SOFT_GATE_THRESHOLDS["hard"]),
    }
    if not raw:
        return out
    for k in ("global", "easy", "medium", "hard"):
        if k not in raw or not isinstance(raw[k], dict):
            continue
        if "conf" in raw[k]:
            try:
                out[k]["conf"] = float(raw[k]["conf"])
            except Exception:
                pass
        if "issues" in raw[k]:
            try:
                out[k]["issues"] = int(raw[k]["issues"])
            except Exception:
                pass
    for k in out:
        out[k]["conf"] = max(0.0, min(1.0, float(out[k]["conf"])))
        out[k]["issues"] = max(1, int(out[k]["issues"]))
    return out


def _compute_data_driven_thresholds_from_history(
    history_files: list[str] | None,
    mode: str,
) -> dict[str, dict[str, float]]:
    thresholds = _coerce_thresholds(None)
    if mode == "off" or not history_files:
        return thresholds

    rows: list[dict[str, Any]] = []
    for fp in history_files:
        path = Path(fp)
        if not path.exists():
            continue
        try:
            import pandas as pd

            df = pd.read_excel(path, sheet_name="Results")
        except Exception:
            continue

        if "ma_confidence" not in df.columns or "ma_critic_issue_count" not in df.columns or "ma_score" not in df.columns:
            continue
        for _, r in df.iterrows():
            try:
                conf = float(r["ma_confidence"])
            except Exception:
                continue
            try:
                issues = int(r["ma_critic_issue_count"])
            except Exception:
                issues = 0
            try:
                score = int(r["ma_score"])
            except Exception:
                continue
            diff = str(r.get("complexity", r.get("Difficulty", ""))).strip().lower()
            if diff not in {"easy", "medium", "hard"}:
                diff = "easy"
            rows.append({"conf": conf, "issues": issues, "score": score, "difficulty": diff})

    if not rows:
        return thresholds

    def derive(subset: list[dict[str, Any]], fallback: dict[str, float]) -> dict[str, float]:
        bad = [x for x in subset if x["score"] <= 1]
        if not bad:
            return dict(fallback)
        bad_conf = sorted(float(x["conf"]) for x in bad)
        bad_issues = sorted(int(x["issues"]) for x in bad)
        # 75th percentile of bad-confidence and median bad-issues.
        idx_conf = min(len(bad_conf) - 1, int(round(0.75 * (len(bad_conf) - 1))))
        idx_issue = min(len(bad_issues) - 1, int(round(0.50 * (len(bad_issues) - 1))))
        conf_th = max(0.35, min(0.9, bad_conf[idx_conf]))
        issue_th = max(1, min(12, bad_issues[idx_issue]))
        return {"conf": float(conf_th), "issues": int(issue_th)}

    global_thr = derive(rows, thresholds["global"])
    thresholds["global"] = global_thr

    if mode == "stratified":
        for d in ("easy", "medium", "hard"):
            subset = [x for x in rows if x["difficulty"] == d]
            thresholds[d] = derive(subset, global_thr)
    return thresholds


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
    question_difficulty: str | None = None,
    soft_gate_conf_threshold: float = 0.58,
    soft_gate_issue_threshold: int = 3,
    soft_gate_stratified_thresholds: bool = False,
    soft_gate_data_mode: str = "off",
    soft_gate_history_files: list[str] | None = None,
    soft_gate_threshold_overrides: dict[str, dict[str, float]] | None = None,
) -> dict:
    start = time.time()
    log_event(
        "multi_agent.start",
        architecture="orchestrator",
        model=model,
        critic_strategy_requested=critic_strategy,
        question_preview=(question or "")[:180],
        question_difficulty=question_difficulty,
        soft_gate_data_mode=soft_gate_data_mode,
    )
    requested_strategy = _normalize_critic_strategy(critic_strategy)
    strategy = requested_strategy
    strategy_note = requested_strategy
    if requested_strategy == "auto":
        strategy, strategy_note = _infer_auto_critic_strategy(question)

    difficulty = _normalize_difficulty(question_difficulty, question)
    data_mode = _normalize_data_mode(soft_gate_data_mode)
    base_thresholds = _coerce_thresholds(soft_gate_threshold_overrides)
    base_thresholds["global"]["conf"] = max(0.0, min(1.0, float(soft_gate_conf_threshold)))
    base_thresholds["global"]["issues"] = max(1, int(soft_gate_issue_threshold))
    if data_mode != "off":
        thresholds = _compute_data_driven_thresholds_from_history(soft_gate_history_files, data_mode)
        # Keep manual global overrides as fallback floor/ceiling if explicitly provided.
        if data_mode == "global":
            thresholds["global"] = dict(thresholds["global"])
        else:
            for d in ("easy", "medium", "hard"):
                thresholds[d] = dict(thresholds[d])
    else:
        thresholds = base_thresholds

    if soft_gate_stratified_thresholds or data_mode == "stratified":
        gate_conf_threshold = float(thresholds.get(difficulty, thresholds["global"])["conf"])
        gate_issue_threshold = int(thresholds.get(difficulty, thresholds["global"])["issues"])
        gate_mode = f"stratified:{difficulty}"
    else:
        gate_conf_threshold = float(thresholds["global"]["conf"])
        gate_issue_threshold = int(thresholds["global"]["issues"])
        gate_mode = "global"

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
        gate_triggered = conf_a < gate_conf_threshold or len(issues_a) >= gate_issue_threshold
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
        "question_difficulty": difficulty,
        "soft_gate_mode": gate_mode,
        "soft_gate_conf_threshold": gate_conf_threshold,
        "soft_gate_issue_threshold": gate_issue_threshold,
        "soft_gate_data_mode": data_mode,
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
        question_difficulty=difficulty,
        soft_gate_mode=gate_mode,
        soft_gate_conf_threshold=gate_conf_threshold,
        soft_gate_issue_threshold=gate_issue_threshold,
        soft_gate_data_mode=data_mode,
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
    question_difficulty: str | None = None,
    soft_gate_conf_threshold: float = 0.58,
    soft_gate_issue_threshold: int = 3,
    soft_gate_stratified_thresholds: bool = False,
    soft_gate_data_mode: str = "off",
    soft_gate_history_files: list[str] | None = None,
    soft_gate_threshold_overrides: dict[str, dict[str, float]] | None = None,
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
            question_difficulty=question_difficulty,
            soft_gate_conf_threshold=soft_gate_conf_threshold,
            soft_gate_issue_threshold=soft_gate_issue_threshold,
            soft_gate_stratified_thresholds=soft_gate_stratified_thresholds,
            soft_gate_data_mode=soft_gate_data_mode,
            soft_gate_history_files=soft_gate_history_files,
            soft_gate_threshold_overrides=soft_gate_threshold_overrides,
        )
    if arch in {"pipeline", "sequential", "sequential-pipeline"}:
        return run_multi_agent_pipeline(client, model, tool_functions, question, verbose=verbose)
    if arch in {"parallel", "parallel-specialists-aggregator"}:
        return run_multi_agent_parallel(client, model, tool_functions, question, verbose=verbose)
    raise ValueError(
        "Unknown multi-agent architecture. Use one of: orchestrator, pipeline, parallel."
    )
