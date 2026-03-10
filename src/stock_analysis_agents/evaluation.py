from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from openai import OpenAI

from .baseline import run_baseline
from .evaluator import run_evaluator
from .models import AgentResult
from .multi_agent import run_multi_agent
from .single_agent import run_single_agent


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



def run_full_evaluation(
    client: OpenAI,
    model: str,
    questions: list[dict],
    tool_functions: dict,
    output_xlsx: str = "results.xlsx",
) -> str:
    records: list[EvalRecord] = []

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
        rec.bl_time = round(time.time() - t0, 2)
        rec.bl_score = run_evaluator(client, model, q["question"], q["expected"], bl.answer).get("score", -1)

        t0 = time.time()
        sa = run_single_agent(client, model, tool_functions, q["question"], verbose=False)
        rec.sa_time = round(time.time() - t0, 2)
        rec.sa_score = run_evaluator(client, model, q["question"], q["expected"], sa.answer).get("score", -1)

        t0 = time.time()
        ma = run_multi_agent(client, model, tool_functions, q["question"], verbose=False)
        rec.ma_time = round(time.time() - t0, 2)
        rec.ma_score = run_evaluator(client, model, q["question"], q["expected"], ma.get("final_answer", "")).get("score", -1)

        records.append(rec)

    df = pd.DataFrame([r.__dict__ for r in records])
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return output_xlsx
