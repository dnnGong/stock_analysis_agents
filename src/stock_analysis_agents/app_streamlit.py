from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Make local SDK importable without requiring global install.
ROOT = Path(__file__).resolve().parent
SDK_SRC = ROOT / "stock_analysis_agents" / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

from stock_analysis_agents import (  # noqa: E402
    FinanceTools,
    build_tool_function_map,
    load_settings,
    make_client,
    make_data_provider,
    run_multi_agent,
    run_single_agent,
)


st.set_page_config(page_title="Stock Analysis Agents", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner=False)
def _init_runtime() -> tuple[Any, Any, dict[str, Any], str]:
    settings = load_settings()
    client = make_client(settings)
    provider = make_data_provider(settings.data_provider, settings.alphavantage_api_key)
    tools = FinanceTools(provider=provider, db_path=settings.db_path)
    tool_map = build_tool_function_map(tools)
    return client, settings, tool_map, provider.name


def _format_history_for_prompt(messages: list[dict[str, Any]], max_msgs: int = 8) -> str:
    recent = messages[-max_msgs:]
    lines: list[str] = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = (msg.get("content") or "").strip()
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _rewrite_followup(client: Any, model: str, history: str, latest_user_msg: str) -> str:
    """Convert follow-up question into a standalone question using chat history."""
    if not history.strip():
        return latest_user_msg

    prompt = (
        "Rewrite the user's latest message into a standalone finance question. "
        "Resolve references like 'that', 'it', 'the two' based on history. "
        "Return only the rewritten standalone question."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history}\n\n"
                        f"Latest user message:\n{latest_user_msg}"
                    ),
                },
            ],
            temperature=0,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or latest_user_msg
    except Exception:
        return latest_user_msg


def _run_agent_turn(
    client: Any,
    tool_map: dict[str, Any],
    architecture: str,
    model: str,
    user_msg: str,
    history_messages: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    history = _format_history_for_prompt(history_messages, max_msgs=8)
    standalone_q = _rewrite_followup(client, model, history, user_msg)

    # Core requirement: pass conversation history on every turn.
    task = (
        "Use the conversation context to answer the latest user question accurately.\n\n"
        f"Conversation context:\n{history}\n\n"
        f"Latest user question:\n{user_msg}\n\n"
        f"Standalone interpreted question:\n{standalone_q}"
    )

    if architecture == "Single Agent":
        out = run_single_agent(client, model, tool_map, task, verbose=False)
        answer = out.answer
        meta = {
            "architecture": "single-agent",
            "model": model,
            "tools": out.tools_called,
            "resolved_question": standalone_q,
        }
        return answer, meta

    out = run_multi_agent(client, model, tool_map, task, verbose=False, architecture="orchestrator")
    answer = out.get("final_answer", "")
    tool_names: list[str] = []
    for r in out.get("agent_results", []):
        tool_names.extend(getattr(r, "tools_called", []))
    meta = {
        "architecture": out.get("architecture", "orchestrator-specialists-critic"),
        "model": model,
        "tools": list(dict.fromkeys(tool_names)),
        "resolved_question": standalone_q,
    }
    return answer, meta


st.title("📈 Stock Analysis Agents")
st.caption("Chat interface for single-agent and multi-agent stock analysis")

# Sidebar controls (assignment requirement)
st.sidebar.header("Controls")
arch_choice = st.sidebar.selectbox("Agent selector", ["Single Agent", "Multi-Agent"], index=1)
model_choice = st.sidebar.selectbox("Model selector", ["gpt-4o-mini", "gpt-4o"], index=0)

if st.sidebar.button("Clear conversation", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

try:
    client, settings, tool_map, provider_name = _init_runtime()
except Exception as exc:
    st.error(f"Initialization failed: {exc}")
    st.stop()

st.sidebar.caption(f"Data provider: {provider_name}")
st.sidebar.caption(f"DB path: {settings.db_path}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display full conversation history (assignment requirement)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta:
            st.caption(
                f"architecture={meta.get('architecture')} | model={meta.get('model')} | "
                f"tools={', '.join(meta.get('tools', [])) or 'none'}"
            )

user_input = st.chat_input("Ask a stock analysis question...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, meta = _run_agent_turn(
                    client=client,
                    tool_map=tool_map,
                    architecture=arch_choice,
                    model=model_choice,
                    user_msg=user_input,
                    history_messages=st.session_state.messages,
                )
            except Exception as exc:
                answer = f"Error: {exc}"
                meta = {
                    "architecture": arch_choice.lower().replace(" ", "-"),
                    "model": model_choice,
                    "tools": [],
                    "resolved_question": user_input,
                }

        st.markdown(answer)
        st.caption(
            f"architecture={meta.get('architecture')} | model={meta.get('model')} | "
            f"tools={', '.join(meta.get('tools', [])) or 'none'}"
        )

        with st.expander("Resolved question", expanded=False):
            st.write(meta.get("resolved_question", ""))

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
