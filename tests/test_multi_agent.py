import pytest

from stock_analysis_agents import multi_agent


def test_normalize_critic_strategy_supports_aliases():
    assert multi_agent._normalize_critic_strategy("strict") == "strict-rewrite"
    assert multi_agent._normalize_critic_strategy("soft_gate") == "soft-gated"
    assert multi_agent._normalize_critic_strategy("dual") == "dual-draft"


def test_normalize_critic_strategy_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown critic strategy"):
        multi_agent._normalize_critic_strategy("mystery")


def test_infer_auto_critic_strategy_uses_complexity():
    strategy, note = multi_agent._infer_auto_critic_strategy(
        "For the top 3 semiconductor stocks by 1-year return, what are their P/E ratios and current news sentiment?"
    )

    assert strategy == "no-rewrite"
    assert note.startswith("auto:no-rewrite")


def test_coerce_thresholds_clamps_values():
    out = multi_agent._coerce_thresholds(
        {
            "global": {"conf": 1.5, "issues": 0},
            "easy": {"conf": -1, "issues": 2.7},
        }
    )

    assert out["global"] == {"conf": 1.0, "issues": 1}
    assert out["easy"] == {"conf": 0.0, "issues": 2}


def test_normalize_difficulty_falls_back_from_question_text():
    assert multi_agent._normalize_difficulty(None, "Compare TSLA sentiment over 6-month performance") == "medium"
    assert multi_agent._normalize_difficulty(None, "Which stocks rose and fell?") == "hard"
    assert multi_agent._normalize_difficulty(None, "What is Apple's P/E ratio?") == "easy"


def test_pack_specialist_outputs_includes_confidence_and_issues():
    specialist = multi_agent.AgentResult(
        agent_name="Market Specialist",
        answer="AAPL rose 10%.",
        tools_called=["get_price_performance"],
        confidence=0.8,
        issues_found=["missing volume"],
    )

    packed = multi_agent._pack_specialist_outputs([specialist], include_confidence_issues=True)

    assert "Confidence: 80%" in packed
    assert "Issues: missing volume" in packed
    assert "Tools: ['get_price_performance']" in packed
