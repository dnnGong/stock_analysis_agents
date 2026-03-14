from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from openai import OpenAI

from .baseline import run_baseline
from .evaluator import run_evaluator
from .multi_agent import run_multi_agent
from .single_agent import run_single_agent
from .structured_logging import log_event


@dataclass
class EvalRecord:
    question_id: str
    question: str
    complexity: str
    category: str
    expected: str
    bl_score: int = -1
    sa_score: int = -1
    ma_score: int = -1
    bl_time: float = 0.0
    sa_time: float = 0.0
    ma_time: float = 0.0
    ma_confidence: float = float("nan")
    ma_critic_issue_count: int = 0
    ma_rewrite_applied: bool = False
    ma_gate_triggered: str = ""
    ma_draft_choice: str = ""
    ma_critic_strategy_requested: str = ""
    ma_critic_strategy_effective: str = ""
    ma_critic_score_a: float = float("nan")
    ma_critic_score_b: float = float("nan")
    ma_calibration_abs_error: float = float("nan")


def _to_float_or_nan(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        v = float(value)
        if math.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def _build_summary_sheet(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arch, score_col, time_col in [
        ("Baseline", "bl_score", "bl_time"),
        ("Single Agent", "sa_score", "sa_time"),
        ("Multi Agent", "ma_score", "ma_time"),
    ]:
        for tier in ["easy", "medium", "hard", "all"]:
            subset = df if tier == "all" else df[df["complexity"] == tier]
            valid = subset[subset[score_col] >= 0]
            avg_score = float(valid[score_col].mean()) if len(valid) else 0.0
            rows.append(
                {
                    "architecture": arch,
                    "difficulty": tier,
                    "questions": int(len(valid)),
                    "avg_score_3": round(avg_score, 3),
                    "accuracy_pct": round(avg_score / 3.0 * 100.0, 1),
                    "avg_time_sec": round(float(subset[time_col].mean()) if len(subset) else 0.0, 3),
                }
            )
    return pd.DataFrame(rows)


def _build_calibration_sheet(df: pd.DataFrame) -> pd.DataFrame:
    # Only MA currently exports explicit confidence in this evaluator.
    conf = pd.to_numeric(df.get("ma_confidence"), errors="coerce")
    score_norm = pd.to_numeric(df.get("ma_score"), errors="coerce") / 3.0
    valid = pd.DataFrame({"conf": conf, "score_norm": score_norm}).dropna()

    rows: list[dict[str, object]] = []
    if len(valid) == 0:
        rows.append(
            {
                "segment": "overall",
                "count": 0,
                "mean_conf": float("nan"),
                "mean_score_norm": float("nan"),
                "mae_conf_vs_score": float("nan"),
                "pearson_conf_score": float("nan"),
            }
        )
        return pd.DataFrame(rows)

    mae = float((valid["conf"] - valid["score_norm"]).abs().mean())
    corr = float(valid["conf"].corr(valid["score_norm"])) if len(valid) > 1 else float("nan")
    rows.append(
        {
            "segment": "overall",
            "count": int(len(valid)),
            "mean_conf": round(float(valid["conf"].mean()), 4),
            "mean_score_norm": round(float(valid["score_norm"].mean()), 4),
            "mae_conf_vs_score": round(mae, 4),
            "pearson_conf_score": round(corr, 4) if math.isfinite(corr) else float("nan"),
        }
    )

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    valid = valid.copy()
    valid["conf_bin"] = pd.cut(valid["conf"], bins=bins, labels=labels, include_lowest=True)
    for label in labels:
        seg = valid[valid["conf_bin"] == label]
        if len(seg) == 0:
            continue
        rows.append(
            {
                "segment": f"bin:{label}",
                "count": int(len(seg)),
                "mean_conf": round(float(seg["conf"].mean()), 4),
                "mean_score_norm": round(float(seg["score_norm"].mean()), 4),
                "mae_conf_vs_score": round(float((seg["conf"] - seg["score_norm"]).abs().mean()), 4),
                "pearson_conf_score": float("nan"),
            }
        )

    return pd.DataFrame(rows)


def run_full_evaluation(
    client: OpenAI,
    model: str,
    questions: list[dict],
    tool_functions: dict,
    output_xlsx: str = "results.xlsx",
    multi_architecture: str = "orchestrator",
    critic_strategy: str = "strict-rewrite",
    soft_gate_conf_threshold: float = 0.58,
    soft_gate_issue_threshold: int = 3,
    soft_gate_stratified_thresholds: bool = False,
    soft_gate_data_mode: str = "off",
    soft_gate_history_files: list[str] | None = None,
    run_config_path: str | None = None,
) -> str:
    records: list[EvalRecord] = []
    log_event(
        "evaluation.start",
        model=model,
        total_questions=len(questions),
        multi_architecture=multi_architecture,
        critic_strategy=critic_strategy,
        output_xlsx=output_xlsx,
    )

    for q in questions:
        rec = EvalRecord(
            question_id=q["id"],
            question=q["question"],
            complexity=q["complexity"],
            category=q["category"],
            expected=q["expected"],
        )

        t0 = time.time()
        bl = run_baseline(client, model, q["question"])
        rec.bl_time = round(time.time() - t0, 3)
        rec.bl_score = run_evaluator(client, model, q["question"], q["expected"], bl.answer).get("score", -1)

        t0 = time.time()
        sa = run_single_agent(client, model, tool_functions, q["question"], verbose=False)
        rec.sa_time = round(time.time() - t0, 3)
        rec.sa_score = run_evaluator(client, model, q["question"], q["expected"], sa.answer).get("score", -1)

        t0 = time.time()
        ma = run_multi_agent(
            client,
            model,
            tool_functions,
            q["question"],
            verbose=False,
            architecture=multi_architecture,
            critic_strategy=critic_strategy,
            question_difficulty=q.get("complexity"),
            soft_gate_conf_threshold=soft_gate_conf_threshold,
            soft_gate_issue_threshold=soft_gate_issue_threshold,
            soft_gate_stratified_thresholds=soft_gate_stratified_thresholds,
            soft_gate_data_mode=soft_gate_data_mode,
            soft_gate_history_files=soft_gate_history_files,
        )
        rec.ma_time = round(time.time() - t0, 3)
        rec.ma_score = run_evaluator(client, model, q["question"], q["expected"], ma.get("final_answer", "")).get("score", -1)

        # Pull diagnostics from orchestrator output when available.
        diag = ma.get("diagnostics", {}) if isinstance(ma.get("diagnostics", {}), dict) else {}
        rec.ma_confidence = _to_float_or_nan(diag.get("critic_confidence"))
        rec.ma_critic_issue_count = int(diag.get("critic_issue_count", 0) or 0)
        rec.ma_rewrite_applied = bool(diag.get("rewrite_applied", False))
        gate = diag.get("gate_triggered", "")
        rec.ma_gate_triggered = "" if gate is None else str(gate)
        rec.ma_draft_choice = str(diag.get("draft_choice", ""))
        rec.ma_critic_strategy_requested = str(diag.get("critic_strategy_requested", ""))
        rec.ma_critic_strategy_effective = str(diag.get("critic_strategy_effective", ""))
        rec.ma_critic_score_a = _to_float_or_nan(diag.get("critic_score_a"))
        rec.ma_critic_score_b = _to_float_or_nan(diag.get("critic_score_b"))

        if math.isfinite(rec.ma_confidence) and rec.ma_score >= 0:
            rec.ma_calibration_abs_error = abs(rec.ma_confidence - (rec.ma_score / 3.0))

        records.append(rec)
        log_event(
            "evaluation.question",
            question_id=rec.question_id,
            difficulty=rec.complexity,
            category=rec.category,
            bl_score=rec.bl_score,
            sa_score=rec.sa_score,
            ma_score=rec.ma_score,
            bl_time=rec.bl_time,
            sa_time=rec.sa_time,
            ma_time=rec.ma_time,
            ma_confidence=rec.ma_confidence,
            ma_critic_issue_count=rec.ma_critic_issue_count,
            ma_strategy_effective=rec.ma_critic_strategy_effective,
            ma_rewrite_applied=rec.ma_rewrite_applied,
            ma_calibration_abs_error=rec.ma_calibration_abs_error,
        )

    df = pd.DataFrame([r.__dict__ for r in records])
    summary = _build_summary_sheet(df)
    calibration = _build_calibration_sheet(df)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        summary.to_excel(writer, index=False, sheet_name="Summary")
        calibration.to_excel(writer, index=False, sheet_name="Calibration")

    if run_config_path is None:
        stem, _ = os.path.splitext(output_xlsx)
        run_config_path = f"{stem}_run_config.json"
    run_config_dir = os.path.dirname(os.path.abspath(run_config_path))
    if run_config_dir:
        os.makedirs(run_config_dir, exist_ok=True)
    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "question_count": len(questions),
        "question_ids": [q.get("id") for q in questions],
        "output_xlsx": output_xlsx,
        "multi_architecture": multi_architecture,
        "critic_strategy": critic_strategy,
        "soft_gate_conf_threshold": soft_gate_conf_threshold,
        "soft_gate_issue_threshold": soft_gate_issue_threshold,
        "soft_gate_stratified_thresholds": soft_gate_stratified_thresholds,
        "soft_gate_data_mode": soft_gate_data_mode,
        "soft_gate_history_files": list(soft_gate_history_files or []),
    }
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    log_event(
        "evaluation.end",
        output_xlsx=output_xlsx,
        run_config_path=run_config_path,
        baseline_avg=float(df["bl_score"].mean()) if len(df) else float("nan"),
        single_avg=float(df["sa_score"].mean()) if len(df) else float("nan"),
        multi_avg=float(df["ma_score"].mean()) if len(df) else float("nan"),
    )
    return output_xlsx
