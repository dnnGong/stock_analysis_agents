from types import SimpleNamespace

from stock_analysis_agents.agent_runner import run_specialist_agent


def _message(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_run_specialist_agent_executes_tool_calls_and_returns_final_answer():
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="echo_tool", arguments='{"ticker": "AAPL"}'),
    )
    responses = [
        _response(_message(tool_calls=[tool_call])),
        _response(_message(content="Final answer with evidence.")),
    ]

    class FakeClient:
        def __init__(self, queued):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            self._queued = list(queued)

        def create(self, **kwargs):
            return self._queued.pop(0)

    client = FakeClient(responses)
    tool_functions = {"echo_tool": lambda ticker: {"ticker": ticker, "ok": True}}

    out = run_specialist_agent(
        client=client,
        model="test-model",
        tool_functions=tool_functions,
        agent_name="Test Agent",
        system_prompt="system",
        task="task",
        tool_schemas=[],
    )

    assert out.answer == "Final answer with evidence."
    assert out.tools_called == ["echo_tool"]
    assert out.raw_data["echo_tool:1"] == {"ticker": "AAPL", "ok": True}


def test_run_specialist_agent_handles_unknown_tool():
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="missing_tool", arguments="{}"),
    )
    responses = [
        _response(_message(tool_calls=[tool_call])),
        _response(_message(content="Done.")),
    ]

    class FakeClient:
        def __init__(self, queued):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            self._queued = list(queued)

        def create(self, **kwargs):
            return self._queued.pop(0)

    out = run_specialist_agent(
        client=FakeClient(responses),
        model="test-model",
        tool_functions={},
        agent_name="Test Agent",
        system_prompt="system",
        task="task",
        tool_schemas=[],
    )

    assert out.answer == "Done."
    assert out.raw_data["missing_tool:1"] == {"error": "Tool missing_tool not found."}


def test_run_specialist_agent_returns_fallback_when_no_final_answer():
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="echo_tool", arguments="{not json"),
    )
    response = _response(_message(tool_calls=[tool_call]))

    class FakeClient:
        def __init__(self, queued):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            self._queued = list(queued)

        def create(self, **kwargs):
            return self._queued.pop(0)

    out = run_specialist_agent(
        client=FakeClient([response]),
        model="test-model",
        tool_functions={"echo_tool": lambda **kwargs: {"kwargs": kwargs}},
        agent_name="Test Agent",
        system_prompt="system",
        task="task",
        tool_schemas=[],
        max_iters=1,
    )

    assert out.answer == "Agent failed to provide an answer."
    assert out.raw_data["echo_tool:1"] == {"kwargs": {}}
