from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    agent_name: str
    answer: str
    tools_called: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    issues_found: list[str] = field(default_factory=list)
    reasoning: str = ""
