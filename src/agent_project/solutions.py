"""The live solution registry and the tools used to operate one solution.

The catalogue belongs to the workspace.  This module keeps no S- or B-number snapshot: every
turn reads ``list_solutions`` and works from the entries that response supplied.  An engine that
answers ``unknown_capability`` predates the registry and must use the methodology path instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from agent_project.node import Node

READS = {
    "get_solution": "Read one live solution entry. Args: solution.",
    "get_policy": "Read the current decision policy. Args come from the workspace contract.",
    "get_experiment": "Read holdout assignment and balance. Args come from the workspace contract.",
    "list_ledger": "Read the solution's action and outcome evidence. Args are optional filters.",
    "get_monitor": "Read drift and guardrail state. Args come from the workspace contract.",
    "get_roi_readout": "Read the measured solution readout. Args come from the workspace contract.",
}

ACTIONS = {
    "create_solution": (
        "Build a catalogue solution. Args: project, solution, level (1-4), "
        "sources {source, table}."
    ),
    "run_solution": "Run a built solution graph. Args: project, graph.",
    "decide": "Apply an approved policy to scores. Args come from the workspace contract.",
    "approve_decision": (
        "Approve one pending A1 decision after explicit owner approval. "
        "Args come from the workspace contract."
    ),
}


def _error_code(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    if isinstance(error, dict):
        return error.get("code")
    code = value.get("code")
    return code if isinstance(code, str) else None


def _http_error_code(error: httpx.HTTPStatusError) -> str | None:
    try:
        return _error_code(error.response.json())
    except ValueError:
        return None


@dataclass(frozen=True)
class Registry:
    """The solution entries observed at the beginning of one turn."""

    entries: list[dict[str, Any]]
    old_engine: bool = False
    failure: str | None = None

    def entry(self, identifier: str | None) -> dict[str, Any] | None:
        return next((item for item in self.entries if item.get("id") == identifier), None)

    def public(self) -> dict[str, Any]:
        value: dict[str, Any] = {"solutions": self.entries}
        if self.old_engine:
            value["old_engine"] = True
        if self.failure:
            value["failure"] = self.failure
        return value

    def observation(self) -> str:
        if self.old_engine:
            return (
                "list_solutions returned unknown_capability. This workspace uses an older "
                "engine, so solution operations are unavailable; use methodologies only."
            )
        if self.failure:
            return f"list_solutions FAILED: {self.failure}"
        return f"list_solutions returned: {self.public()}"


def read(node: Node | None) -> Registry:
    """Read this turn's solution registry without inventing an empty catalogue."""
    if node is None:
        return Registry([], failure="OFFLINE: no workspace read ran")
    try:
        value = node.read("list_solutions")
    except httpx.HTTPStatusError as error:
        if _http_error_code(error) == "unknown_capability":
            return Registry([], old_engine=True)
        return Registry([], failure=f"{error.response.status_code} {error.response.text[:200]}")
    except httpx.HTTPError as error:
        return Registry([], failure=f"{type(error).__name__}: {error}")

    if _error_code(value) == "unknown_capability":
        return Registry([], old_engine=True)
    if not isinstance(value, dict) or not isinstance(value.get("solutions"), list):
        return Registry([], failure=f"unexpected response {value!r}")
    entries = [item for item in value["solutions"] if isinstance(item, dict)]
    return Registry(entries)


def specs() -> list[dict[str, Any]]:
    """Solution tools in the OpenAI SDK shape; policy mutation is intentionally absent."""
    result = []
    for name, description in {**READS, **ACTIONS}.items():
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "args": {
                                "type": "object",
                                "description": "Arguments named in the capability description.",
                            },
                            "because": {
                                "type": "string",
                                "description": (
                                    "Quote the previous observation that justifies this call."
                                ),
                            },
                        },
                        "required": ["args", "because"],
                    },
                },
            }
        )
    return result
