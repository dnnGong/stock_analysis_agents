from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from .models import AgentResult



def run_specialist_agent(
    client: OpenAI,
    model: str,
    tool_functions: dict[str, Any],
    agent_name: str,
    system_prompt: str,
    task: str,
    tool_schemas: list[dict],
    max_iters: int = 8,
    max_tool_calls_per_turn: int = 40,
    verbose: bool = False,
) -> AgentResult:
    start_time = time.time()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    tools_called: list[str] = []
    raw_data: dict[str, Any] = {}
    iterations = 0
    final_answer = ""

    while iterations < max_iters:
        iterations += 1

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            temperature=0,
        )

        msg = response.choices[0].message
        tool_calls = list(msg.tool_calls or [])

        if tool_calls:
            if len(tool_calls) > max_tool_calls_per_turn:
                tool_calls = tool_calls[:max_tool_calls_per_turn]

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments or "{}")
                except Exception:
                    func_args = {}

                if verbose:
                    print(f"[{agent_name}] Calling: {func_name}({func_args})")

                if func_name not in tool_functions:
                    tool_out = {"error": f"Tool {func_name} not found."}
                else:
                    try:
                        tool_out = tool_functions[func_name](**func_args)
                    except Exception as exc:
                        tool_out = {"error": str(exc)}

                tools_called.append(func_name)
                raw_data[f"{func_name}:{len(tools_called)}"] = tool_out

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": json.dumps(tool_out),
                    }
                )
            continue

        messages.append({"role": "assistant", "content": msg.content or ""})
        final_answer = msg.content or ""
        break

    duration = time.time() - start_time
    if not final_answer:
        final_answer = "Agent failed to provide an answer."

    return AgentResult(
        agent_name=agent_name,
        answer=final_answer,
        tools_called=tools_called,
        raw_data=raw_data,
        reasoning=f"Completed in {iterations} iterations. Time taken: {duration:.2f}s",
    )
