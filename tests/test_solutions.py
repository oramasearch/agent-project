"""Solution discovery uses the workspace registry and has one compatibility fallback."""

from __future__ import annotations

import httpx
import json
from types import SimpleNamespace

from agent_project import service, solutions
from agent_project.node import ACT_CAPABILITY, READ_CAPABILITY
from agent_project.retrieval import Corpus


class FakeNode:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def read(self, capability, **params):
        self.calls.append((capability, params))
        return self.value


def completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    message.model_dump = lambda exclude_none=True: {"role": "assistant", "content": content or ""}
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def call(name, args, because):
    payload = {"args": args, "because": because}
    return SimpleNamespace(
        id=f"call-{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(payload)),
    )


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_):
        return self.replies.pop(0)


def solution_entry(runnable=True, missing=None, status="available"):
    return {
        "id": "S-live",
        "name": "Retention actions",
        "promise": "Prioritize retention work",
        "use_case": "retention",
        "composition": [{"methodology": "M-1", "minimum_level": 1}],
        "levels": [{"level": 1, "name": "Assist"}],
        "action_patterns": ["task"],
        "latency": ["batch"],
        "autonomy": [{"action_class": "task", "autonomy": "A1"}],
        "data_roles": [{"role": "outcome", "required": True,
                        "description": "Observed retention outcome"}],
        "policy": "Act above an owner-approved threshold",
        "measurement": "Compare treatment with the holdout",
        "guardrails": ["contact cap"],
        "runnable": runnable,
        "status": status,
        "missing": missing or [],
        "issue": "zetta#solution",
        "promotion_condition": "Implement the foundation callbacks",
    }


class SolutionNode:
    def __init__(self, entry):
        self.entry = entry
        self.calls = []

    def read(self, capability, **params):
        self.calls.append(("read", capability, params))
        if capability == "list_solutions":
            return {"solutions": [self.entry]}
        if capability == "get_solution":
            return {"solution": self.entry}
        if capability == "get_policy":
            return {"policy": {"version": 2, "autonomy": "A1"}}
        if capability == "get_experiment":
            return {"experiment": {"holdout": "configured"}}
        if capability == "list_ledger":
            return {"entries": [{"kind": "action"}]}
        if capability == "get_monitor":
            return {"monitor": {"status": "within_bounds"}}
        if capability == "get_roi_readout":
            return {"readout": {"incrementality": "measured against holdout"}}
        raise AssertionError(capability)

    def act(self, capability, **params):
        self.calls.append(("act", capability, params))
        if capability == "create_solution":
            return {
                "project": params["project"],
                "solution": params["solution"],
                "level": params["level"],
                "graph": "S-live CL-1 preview",
                "preview": not self.entry["runnable"],
                "missing": self.entry["missing"],
            }
        if capability == "run_solution":
            return {"run_id": "run-1", "status": "succeeded", "preview": True}
        raise AssertionError(capability)


def operation_script(final_reply="The preview ran and the holdout readout is available."):
    project = "retention"
    graph = "S-live CL-1 preview"
    return Scripted(
        completion(tool_calls=[call("get_solution", {"solution": "S-live"},
                                    "list_solutions returned S-live")]),
        completion(tool_calls=[call("create_solution", {
            "project": project,
            "solution": "S-live",
            "level": 1,
            "sources": {"source": "datasets", "table": "customers"},
        }, "get_solution returned level 1 and the required outcome role")]),
        completion(tool_calls=[call("run_solution", {"project": project, "graph": graph},
                                    "create_solution returned the preview graph")]),
        completion(tool_calls=[call("get_policy", {"project": project},
                                    "run_solution returned succeeded")]),
        completion(tool_calls=[call("get_experiment", {"project": project},
                                    "get_policy returned version 2")]),
        completion(tool_calls=[call("list_ledger", {"project": project},
                                    "get_experiment returned a configured holdout")]),
        completion(tool_calls=[call("get_monitor", {"project": project},
                                    "list_ledger returned action evidence")]),
        completion(tool_calls=[call("get_roi_readout", {"project": project},
                                    "get_monitor returned within bounds")]),
        completion(content=json.dumps({
            "solution": {"id": "S-live"},
            "reply": final_reply,
        })),
    )


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


def test_a_runnable_solution_builds_runs_and_explains_live_evidence():
    node = SolutionNode(solution_entry())
    registry = solutions.read(node)

    result = solutions.operate(
        {"project": "retention", "intent": "build retention actions"},
        operation_script(),
        "model",
        node,
        registry,
    )

    assert result["kind"] == "solution"
    assert result["solution"]["id"] == "S-live"
    assert result["solution"]["production_ready"] is True
    assert result["policy_proposal"]["approval_required"] is True
    assert result["label_definitions"][0]["role"] == "outcome"
    assert result["owner_signoff"] == {
        "required": True,
        "items": ["policy_proposal", "label_definitions"],
    }
    assert result["operations"]["get_roi_readout"]["value"]["readout"]
    assert result["readout"] == {
        "status": "available",
        "value": {"readout": {"incrementality": "measured against holdout"}},
        "explanation": "The preview ran and the holdout readout is available.",
    }
    assert result["prompt_version"] == "v5"
    assert not solutions.defects(result)
    assert [item[1] for item in node.calls[:3]] == [
        "list_solutions", "get_solution", "create_solution",
    ]


def test_missing_components_are_a_named_production_caveat_not_a_build_refusal():
    missing = [{"id": "readout", "name": "Measured readout", "issue": "zetta#1108"}]
    node = SolutionNode(solution_entry(runnable=False, missing=missing))
    registry = solutions.read(node)

    result = solutions.operate(
        {"project": "retention", "intent": "build retention actions"},
        operation_script("The methodology preview ran."),
        "model",
        node,
        registry,
    )

    assert result["solution"]["preview"] is True
    assert result["solution"]["production_ready"] is False
    assert result["solution"]["missing"] == missing
    assert "zetta#1108" in result["reply"]
    assert not solutions.defects(result)


def test_a_sketch_is_read_then_refused_with_its_promotion_condition():
    entry = solution_entry(runnable=False, status="sketch")
    node = SolutionNode(entry)
    registry = solutions.read(node)
    client = Scripted(
        completion(tool_calls=[call("get_solution", {"solution": "S-live"},
                                    "list_solutions returned S-live as a sketch")]),
        completion(content=json.dumps({
            "declined": {
                "solution": "S-live",
                "reason": entry["promotion_condition"],
            },
            "reply": "This sketch cannot be built.",
        })),
    )

    result = solutions.operate(
        {"project": "retention", "solution": "S-live", "intent": "build it"},
        client,
        "model",
        node,
        registry,
    )

    assert result["kind"] == "declined"
    assert entry["promotion_condition"] in result["reply"]
    assert all(item[0] == "read" for item in node.calls)
    assert not solutions.defects(result)


def test_a_policy_change_is_proposed_for_approval_and_never_applied():
    entry = solution_entry()
    node = SolutionNode(entry)
    registry = solutions.read(node)
    client = Scripted(
        completion(tool_calls=[call("get_solution", {"solution": "S-live"},
                                    "list_solutions returned S-live")]),
        completion(tool_calls=[call("get_policy", {"project": "retention"},
                                    "get_solution returned the A1 policy template")]),
        completion(content=json.dumps({
            "policy_change": {
                "solution": "S-live",
                "before": {"threshold": 0.8},
                "proposed": {"threshold": 0.75},
                "reason": "Review more candidates",
            },
        })),
    )

    result = solutions.operate(
        {"project": "retention", "solution": "S-live", "intent": "change the threshold"},
        client,
        "model",
        node,
        registry,
    )

    assert result["kind"] == "policy_proposal"
    assert result["approval_required"] is True
    assert result["reply"].startswith("Proposed, not applied")
    assert result["policy_change"]["before"] == {
        "policy": {"version": 2, "autonomy": "A1"},
    }
    assert all(item[0] == "read" for item in node.calls)
    assert not solutions.defects(result)


def test_the_service_routes_a_current_engine_turn_to_solution_operation(tmp_path):
    node = SolutionNode(solution_entry())
    settings = service.Settings()
    settings.trace_dir = tmp_path / "traces"

    answer = service.run_turn(
        {"request": "build retention actions", "context": {"project": "retention"}},
        settings,
        Corpus(tmp_path / "missing.jsonl"),
        client=operation_script(),
        node=node,
    )

    assert answer["ok"] is True
    assert answer["result"]["kind"] == "solution"
    assert answer["result"]["defects"] == []
    assert node.calls[0][1] == "list_solutions"
    assert [item for item in node.calls if item[1] == "list_solutions"] == [node.calls[0]]
