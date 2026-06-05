from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import streamlit as st

# Allow `streamlit run src/stock_analysis_agents/app_streamlit.py` from a checkout
# without requiring an editable install first.
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_analysis_agents import (  # noqa: E402
    FinanceTools,
    Settings,
    build_tool_function_map,
    load_settings,
    make_client,
    make_data_provider,
    run_multi_agent,
    run_single_agent,
)


APP_TITLE = "Stock Analysis Agents"
EXAMPLE_QUESTIONS = [
    "Compare the 1-year returns of AAPL, MSFT, and GOOGL.",
    "What is Apple's P/E ratio and how does recent news sentiment look?",
    "Which semiconductor stocks have grown the most over the past year?",
    "Are US markets open right now?",
]
CRITIC_STRATEGIES = [
    "strict-rewrite",
    "no-rewrite",
    "soft-gated",
    "dual-draft",
    "minimal-rewrite",
    "auto",
]
MULTI_ARCHITECTURES = {
    "Orchestrator + Critic": "orchestrator",
    "Sequential Pipeline": "pipeline",
    "Parallel Specialists": "parallel",
}


st.set_page_config(page_title=APP_TITLE, page_icon="ST", layout="wide")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --stock-bg: #f7f8fb;
            --stock-panel: #ffffff;
            --stock-border: #dfe4ec;
            --stock-muted: #667085;
            --stock-ink: #182230;
            --stock-accent: #0f766e;
            --stock-accent-soft: #dff7f2;
        }

        .stApp {
            background: var(--stock-bg);
            color: var(--stock-ink);
        }

        [data-testid="stHeader"] {
            background: rgba(247, 248, 251, 0.9);
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--stock-border);
        }

        .app-shell {
            max-width: 1180px;
            margin: 0 auto;
            padding: 0.25rem 0 2rem;
        }

        .app-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.4rem 0 1rem;
            border-bottom: 1px solid var(--stock-border);
        }

        .app-title h1 {
            font-size: 2rem;
            line-height: 1.15;
            margin: 0;
            letter-spacing: 0;
        }

        .app-title p {
            color: var(--stock-muted);
            margin: 0.35rem 0 0;
            max-width: 720px;
        }

        .status-pill {
            border: 1px solid var(--stock-border);
            border-radius: 999px;
            background: var(--stock-panel);
            color: var(--stock-muted);
            font-size: 0.8rem;
            padding: 0.35rem 0.7rem;
            white-space: nowrap;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0;
        }

        .metric-tile {
            background: var(--stock-panel);
            border: 1px solid var(--stock-border);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            min-height: 82px;
        }

        .metric-tile span {
            display: block;
            color: var(--stock-muted);
            font-size: 0.78rem;
            margin-bottom: 0.35rem;
        }

        .metric-tile strong {
            display: block;
            color: var(--stock-ink);
            font-size: 1rem;
            overflow-wrap: anywhere;
        }

        .example-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.6rem 0 1rem;
        }

        .stButton button {
            border-radius: 7px;
            border-color: var(--stock-border);
        }

        div[data-testid="stChatMessage"] {
            background: var(--stock-panel);
            border: 1px solid var(--stock-border);
            border-radius: 8px;
            padding: 0.85rem;
        }

        div[data-testid="stChatMessage"] + div[data-testid="stChatMessage"] {
            margin-top: 0.7rem;
        }

        @media (max-width: 780px) {
            .app-title {
                align-items: flex-start;
                flex-direction: column;
            }

            .metric-row,
            .example-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_base_settings() -> tuple[Settings | None, str | None]:
    try:
        return load_settings(), None
    except Exception as exc:
        fallback = Settings(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY", "").strip(),
            data_provider=os.getenv("STOCK_AGENTS_DATA_PROVIDER", "yahoo").strip().lower(),
            model_small=os.getenv("STOCK_AGENTS_MODEL_SMALL", "gpt-4o-mini"),
            model_large=os.getenv("STOCK_AGENTS_MODEL_LARGE", "gpt-4o"),
            active_model=os.getenv("STOCK_AGENTS_MODEL", "gpt-4o-mini"),
            db_path=Path(os.getenv("STOCK_AGENTS_DB_PATH", "stocks.db")),
        )
        return fallback, str(exc)


@st.cache_resource(show_spinner=False)
def _init_runtime(
    openai_api_key: str,
    alphavantage_api_key: str,
    provider_name: str,
    model_small: str,
    model_large: str,
    active_model: str,
    db_path: str,
) -> tuple[Any, dict[str, Any], str]:
    settings = Settings(
        openai_api_key=openai_api_key,
        alphavantage_api_key=alphavantage_api_key,
        data_provider=provider_name,
        model_small=model_small,
        model_large=model_large,
        active_model=active_model,
        db_path=Path(db_path),
    )
    client = make_client(settings)
    provider = make_data_provider(settings.data_provider, settings.alphavantage_api_key)
    finance_tools = FinanceTools(provider=provider, db_path=settings.db_path)
    tool_map = build_tool_function_map(finance_tools)
    return client, tool_map, provider.name


def _format_history_for_prompt(messages: list[dict[str, Any]], max_msgs: int = 8) -> str:
    recent = messages[-max_msgs:]
    lines: list[str] = []
    for msg in recent:
        content = (msg.get("content") or "").strip()
        if content:
            lines.append(f"{msg.get('role', 'unknown').upper()}: {content}")
    return "\n".join(lines)


def _rewrite_followup(client: Any, model: str, history: str, latest_user_msg: str) -> str:
    if not history.strip():
        return latest_user_msg

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the latest user message into a standalone finance question. "
                        "Use the conversation history only to resolve references. "
                        "Return only the rewritten question."
                    ),
                },
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
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or latest_user_msg
    except Exception:
        return latest_user_msg


def _tool_names_from_multi_agent(out: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for result in out.get("agent_results", []):
        names.extend(getattr(result, "tools_called", []))
    return list(dict.fromkeys(names))


def _agent_summaries(out: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for result in out.get("agent_results", []):
        summaries.append(
            {
                "agent": getattr(result, "agent_name", "Agent"),
                "confidence": getattr(result, "confidence", 0.0),
                "tools": ", ".join(getattr(result, "tools_called", []) or []) or "none",
                "issues": "; ".join(getattr(result, "issues_found", []) or []) or "none",
            }
        )
    return summaries


def _run_agent_turn(
    client: Any,
    tool_map: dict[str, Any],
    architecture_choice: str,
    multi_architecture: str,
    critic_strategy: str,
    model: str,
    user_msg: str,
    history_messages: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    history = _format_history_for_prompt(history_messages, max_msgs=8)
    standalone_question = _rewrite_followup(client, model, history, user_msg)
    task = (
        "Use the conversation context to answer the latest user question accurately.\n\n"
        f"Conversation context:\n{history}\n\n"
        f"Latest user question:\n{user_msg}\n\n"
        f"Standalone interpreted question:\n{standalone_question}"
    )

    if architecture_choice == "Single Agent":
        out = run_single_agent(client, model, tool_map, task, verbose=False)
        return out.answer, {
            "architecture": "single-agent",
            "model": model,
            "tools": list(dict.fromkeys(out.tools_called)),
            "resolved_question": standalone_question,
            "diagnostics": {},
            "agents": [
                {
                    "agent": out.agent_name,
                    "confidence": out.confidence,
                    "tools": ", ".join(out.tools_called) or "none",
                    "issues": "; ".join(out.issues_found) or "none",
                }
            ],
        }

    out = run_multi_agent(
        client=client,
        model=model,
        tool_functions=tool_map,
        question=task,
        verbose=False,
        architecture=multi_architecture,
        critic_strategy=critic_strategy,
    )
    return str(out.get("final_answer", "")), {
        "architecture": out.get("architecture", multi_architecture),
        "model": model,
        "tools": _tool_names_from_multi_agent(out),
        "resolved_question": standalone_question,
        "diagnostics": out.get("diagnostics", {}),
        "agents": _agent_summaries(out),
        "elapsed_sec": out.get("elapsed_sec"),
    }


def _reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_prompt = ""


def _set_example_prompt(prompt: str) -> None:
    st.session_state.pending_prompt = prompt


def _render_header(provider_name: str, db_path: Path, model: str, architecture: str) -> None:
    st.markdown('<div class="app-shell">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="app-title">
          <div>
            <h1>{APP_TITLE}</h1>
            <p>Interactive workspace for single-agent and multi-agent stock analysis.</p>
          </div>
          <div class="status-pill">Provider: {provider_name}</div>
        </div>
        <div class="metric-row">
          <div class="metric-tile"><span>Architecture</span><strong>{architecture}</strong></div>
          <div class="metric-tile"><span>Model</span><strong>{model}</strong></div>
          <div class="metric-tile"><span>Data Source</span><strong>{provider_name}</strong></div>
          <div class="metric-tile"><span>Local DB</span><strong>{db_path}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_meta(meta: dict[str, Any]) -> None:
    tools = ", ".join(meta.get("tools", [])) or "none"
    elapsed = meta.get("elapsed_sec")
    elapsed_txt = f" | elapsed={elapsed:.2f}s" if isinstance(elapsed, (int, float)) else ""
    st.caption(
        f"architecture={meta.get('architecture')} | model={meta.get('model')} | "
        f"tools={tools}{elapsed_txt}"
    )

    with st.expander("Diagnostics", expanded=False):
        st.write("Resolved question")
        st.code(meta.get("resolved_question", ""), language="text")
        diagnostics = meta.get("diagnostics") or {}
        agents = meta.get("agents") or []
        if diagnostics:
            st.json(diagnostics)
        if agents:
            st.dataframe(agents, use_container_width=True, hide_index=True)


def _render_message(msg: dict[str, Any]) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta")
        if meta:
            _render_meta(meta)


def _sidebar(base_settings: Settings, settings_error: str | None) -> tuple[Settings, str, str, str]:
    st.sidebar.header("Run Settings")
    if settings_error:
        st.sidebar.warning(settings_error)

    provider_options = ["hybrid", "alphavantage", "yahoo"]
    provider_index = (
        provider_options.index(base_settings.data_provider)
        if base_settings.data_provider in provider_options
        else 0
    )
    provider_name = st.sidebar.selectbox("Data provider", provider_options, index=provider_index)

    model_options = list(dict.fromkeys([base_settings.active_model, base_settings.model_small, base_settings.model_large]))
    model = st.sidebar.selectbox("Model", model_options, index=0)
    architecture_choice = st.sidebar.radio("Agent mode", ["Multi-Agent", "Single Agent"], index=0)
    multi_arch_label = st.sidebar.selectbox("Multi-agent pattern", list(MULTI_ARCHITECTURES), index=0)
    critic_strategy = st.sidebar.selectbox("Critic strategy", CRITIC_STRATEGIES, index=5)

    st.sidebar.divider()
    st.sidebar.header("Credentials")
    openai_api_key = st.sidebar.text_input(
        "OpenAI API key",
        value=base_settings.openai_api_key,
        type="password",
        help="Uses OPENAI_API_KEY when available.",
    )
    alphavantage_api_key = st.sidebar.text_input(
        "Alpha Vantage API key",
        value=base_settings.alphavantage_api_key,
        type="password",
        help="Required for alphavantage or hybrid provider modes.",
    )
    db_path = st.sidebar.text_input("stocks.db path", value=str(base_settings.db_path))

    st.sidebar.divider()
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        st.button("Clear", on_click=_reset_chat, use_container_width=True)
    with col_b:
        st.button("Refresh", on_click=st.cache_resource.clear, use_container_width=True)

    settings = replace(
        base_settings,
        openai_api_key=openai_api_key.strip(),
        alphavantage_api_key=alphavantage_api_key.strip(),
        data_provider=provider_name,
        active_model=model,
        db_path=Path(db_path).expanduser(),
    )
    return settings, architecture_choice, MULTI_ARCHITECTURES[multi_arch_label], critic_strategy


def main() -> None:
    _inject_css()

    base_settings, settings_error = _load_base_settings()
    if base_settings is None:
        st.error("Unable to create app settings.")
        st.stop()

    settings, architecture_choice, multi_architecture, critic_strategy = _sidebar(base_settings, settings_error)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""

    provider_label = settings.data_provider
    try:
        client, tool_map, provider_label = _init_runtime(
            openai_api_key=settings.openai_api_key,
            alphavantage_api_key=settings.alphavantage_api_key,
            provider_name=settings.data_provider,
            model_small=settings.model_small,
            model_large=settings.model_large,
            active_model=settings.active_model,
            db_path=str(settings.db_path),
        )
    except Exception as exc:
        _render_header(provider_label, settings.db_path, settings.active_model, architecture_choice)
        st.error(f"Runtime initialization failed: {exc}")
        st.info("Set API keys in the sidebar or export environment variables, then press Refresh.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    _render_header(provider_label, settings.db_path, settings.active_model, architecture_choice)

    if not st.session_state.messages:
        st.subheader("Start with a question")
        st.markdown('<div class="example-grid">', unsafe_allow_html=True)
        for question in EXAMPLE_QUESTIONS:
            st.button(question, on_click=_set_example_prompt, args=(question,), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    for message in st.session_state.messages:
        _render_message(message)

    prompt = st.chat_input("Ask about tickers, sectors, returns, fundamentals, or sentiment...")
    if not prompt and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""

    if prompt:
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        _render_message(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Running stock analysis agents..."):
                try:
                    answer, meta = _run_agent_turn(
                        client=client,
                        tool_map=tool_map,
                        architecture_choice=architecture_choice,
                        multi_architecture=multi_architecture,
                        critic_strategy=critic_strategy,
                        model=settings.active_model,
                        user_msg=prompt,
                        history_messages=st.session_state.messages,
                    )
                except Exception as exc:
                    answer = f"Analysis failed: {exc}"
                    meta = {
                        "architecture": architecture_choice.lower().replace(" ", "-"),
                        "model": settings.active_model,
                        "tools": [],
                        "resolved_question": prompt,
                        "diagnostics": {"error": str(exc)},
                        "agents": [],
                    }

            assistant_message = {"role": "assistant", "content": answer, "meta": meta}
            st.markdown(answer)
            _render_meta(meta)
            st.session_state.messages.append(assistant_message)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
