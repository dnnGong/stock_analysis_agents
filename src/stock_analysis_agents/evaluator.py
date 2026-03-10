from __future__ import annotations

import json

from openai import OpenAI



def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if t.lower().startswith("json\n"):
        t = t[5:].strip()
    return t



def run_evaluator(client: OpenAI, model: str, question: str, expected_answer: str, agent_answer: str) -> dict:
    answer_l = (agent_answer or "").lower()
    refusal_markers = [
        "cannot retrieve",
        "can't retrieve",
        "unable to",
        "i cannot",
        "i can't",
        "please check",
    ]
    if any(m in answer_l for m in refusal_markers):
        return {
            "score": 0,
            "max_score": 3,
            "reasoning": "The answer refuses to provide the requested result.",
            "hallucination_detected": False,
            "key_issues": ["Refusal/no-attempt answer"],
        }

    prompt = """You are a strict evaluator for finance QA answers.
Return STRICT JSON only:
{
  "score": 0,
  "max_score": 3,
  "reasoning": "...",
  "hallucination_detected": false,
  "key_issues": ["..."]
}
Rubric:
3 fully correct; 2 partial; 1 mostly wrong; 0 failure/refusal.
Flag hallucination for unsupported specific numeric claims.
"""

    user_msg = (
        f"Question:\n{question}\n\n"
        f"Expected answer description:\n{expected_answer}\n\n"
        f"Agent answer:\n{agent_answer}\n\n"
        "Return JSON only."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        data = json.loads(_strip_code_fences(resp.choices[0].message.content or ""))
    except Exception:
        return {
            "score": 0,
            "max_score": 3,
            "reasoning": "evaluator parse error",
            "hallucination_detected": False,
            "key_issues": ["evaluator failed to parse"],
        }

    out = {
        "score": data.get("score", 0),
        "max_score": 3,
        "reasoning": str(data.get("reasoning", "")),
        "hallucination_detected": bool(data.get("hallucination_detected", False)),
        "key_issues": data.get("key_issues", []),
    }
    try:
        out["score"] = max(0, min(3, int(out["score"])))
    except Exception:
        out["score"] = 0

    if not isinstance(out["key_issues"], list):
        out["key_issues"] = [str(out["key_issues"])]
    out["key_issues"] = [str(x) for x in out["key_issues"]][:6]
    if not out["reasoning"]:
        out["reasoning"] = "Evaluation completed with limited justification."
    return out
