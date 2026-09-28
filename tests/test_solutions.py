"""Solution discovery uses the workspace registry and has one compatibility fallback."""

from __future__ import annotations

import httpx

from agent_project import solutions
from agent_project.node import ACT_CAPABILITY, READ_CAPABILITY


class FakeNode:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def read(self, capability, **params):
        self.calls.append((capability, params))
        return self.value


def test_the_live_registry_is_kept_verbatim():
    entries = [{
        "id": "S-live",
        "name": "Workspace-owned solution",
        "runnable": True,
        "levels": [{"level": 1}],
    }]
    node = FakeNode({"solutions": entries})

    registry = solutions.read(node)

    assert registry.entries == entries
    assert registry.entry("S-live") is entries[0]
    assert node.calls == [("list_solutions", {})]


def test_unknown_capability_marks_an_old_engine():
    class OldNode(FakeNode):
        def read(self, capability, **params):
            request = httpx.Request("GET", "https://workspace/v1/list_solutions")
            response = httpx.Response(
                404,
                json={"error": {"code": "unknown_capability"}},
                request=request,
            )
            raise httpx.HTTPStatusError("unknown capability", request=request, response=response)

    registry = solutions.read(OldNode({}))

    assert registry.old_engine is True
    assert registry.entries == []
    assert "methodologies only" in registry.observation()


def test_solution_tools_match_workspace_capabilities_without_policy_mutation():
    assert {"list_solutions", "get_solution", "get_policy", "get_experiment",
            "list_ledger", "get_monitor", "get_roi_readout"} <= set(READ_CAPABILITY)
    assert {"create_solution", "run_solution", "decide", "approve_decision"} <= set(
        ACT_CAPABILITY
    )
    names = {item["function"]["name"] for item in solutions.specs()}
    assert "update_policy" not in names
    assert "create_policy" not in names
