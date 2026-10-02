"""Methodology selection is grounded in each workspace's live registry and data plan."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from agent_project import agent, edit
from agent_project.retrieval import Corpus
from test_edit import PIPELINE, turn


def completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    message.model_dump = lambda exclude_none=True: {"role": "assistant", "content": content or ""}
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def build_call(name, args, because):
    payload = {"args": args, "because": because}
    return SimpleNamespace(id=f"call-{name}",
                           function=SimpleNamespace(name=name, arguments=json.dumps(payload)))


def edit_call(name, args):
    return SimpleNamespace(id=f"call-{name}",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        return self.replies.pop(0)


class FakeNode:
    def __init__(self, registry, feature_plan=None):
        self.registry = registry
        self.feature_plan = feature_plan or {"features": ["recency", "frequency"]}
        self.calls = []

    def read(self, capability, **params):
        self.calls.append(("read", capability, params))
        if capability == "list_solutions":
            return {"error": {"code": "unknown_capability"}}
        if capability == "list_methodologies":
            return self.registry
        if capability == "list_datasets":
            return {"datasets": [{"database": "datasets", "name": "customers"}]}
        if capability == "get_dataset":
            return {"columns": ["customer_id", "week", "sales", "event_at"]}
        if capability == "analyze_dataset":
            return {"grain": "snapshot", "keys": ["customer_id"], "leakage_suspects": []}
        if capability == "plan_features":
            return self.feature_plan
        raise AssertionError(capability)

    def act(self, capability, **params):
        self.calls.append(("act", capability, params))
        assert capability == "compile_intent"
        return {"methodology": params["methodology"],
                "nodes": [{"kind": "connect"}, {"kind": "model"}]}


def method(identifier, name, kind, runnable=True, issue=None):
    value = {"id": identifier, "name": name, "family": name.lower(),
             "model_kinds": [kind], "inputs": [], "outputs": [], "metrics": [],
             "runnable": runnable, "partial": False}
    if issue:
        value["issue"] = issue
    return value


def build_script(identifier, model_kind, reply):
    return Scripted(
        completion(tool_calls=[build_call(
            "list_datasets", {}, "list_methodologies returned the runnable methodology")]),
        completion(tool_calls=[build_call(
            "analyze_dataset", {"dataset": "datasets.customers"},
            "list_datasets returned datasets.customers")]),
        completion(tool_calls=[build_call(
            "plan_features", {"dataset": "datasets.customers", "methodology": identifier},
            "analyze_dataset returned snapshot grain")]),
        completion(tool_calls=[build_call(
            "compile_intent", {"project": "customer-project", "intent": reply,
                               "methodology": identifier},
            "plan_features returned recency and frequency")]),
        completion(content=json.dumps({
            "pipeline": {"dataset": "datasets.customers", "methodology": identifier,
                         "model_kind": model_kind, "features": ["recency", "frequency"]},
            "reply": reply,
        })),
    )


@pytest.mark.parametrize(("intent", "identifier", "model_kind"), [
    ("segment customers", "M-3", "kmeans"),
    ("forecast weekly demand", "M-4", "seasonal_forecast"),
])
def test_a_runnable_methodology_is_analyzed_planned_and_compiled(
        tmp_path, monkeypatch, intent, identifier, model_kind):
    monkeypatch.setattr(agent.catalog, "as_context", lambda: "")
    registry = {"methodologies": [method(identifier, intent, model_kind)]}
    node = FakeNode(registry)
    result = agent.build(
        {"project": "customer-project", "profile": {}, "intent": intent},
        Corpus(tmp_path / "missing.jsonl"), build_script(identifier, model_kind, intent),
        "model", node, 0,
    )

    assert result["pipeline"]["methodology"] == identifier
    assert result["pipeline"]["model_kind"] == model_kind
    assert result["prompt_version"] == "v6"
    assert not agent.defects({"intent": intent}, result)
    assert node.calls[0][1] == "list_solutions"
    assert node.calls[1][1] == "list_methodologies"
    # Zetta's compile_intent accepts no `intent`: the model's goal text is not forwarded
    # (agent-project#20); the compiler reads the project's captured intent.
    assert node.calls[-1] == (
        "act", "compile_intent",
        {"project": "customer-project", "methodology": identifier},
    )


def test_an_unsupported_methodology_is_declined_with_the_registry_issue(tmp_path, monkeypatch):
    monkeypatch.setattr(agent.catalog, "as_context", lambda: "")
    node = FakeNode({"methodologies": [
        method("M-3", "Segmentation / clustering", "kmeans", False, "zetta#1042")
    ]})
    client = Scripted(completion(content=json.dumps({
        "pipeline": {"dataset": "datasets.customers", "methodology": "M-3",
                     "model_kind": "kmeans", "features": ["recency"]},
        "reply": "A segmentation pipeline.",
    })))
    task = {"project": "customer-project", "profile": {}, "intent": "segment customers"}

    result = agent.build(task, Corpus(tmp_path / "missing.jsonl"), client, "model", node, 0)

    assert result["pipeline"] is None
    assert result["declined"]["issue"] == "zetta#1042"
    assert "zetta#1042" in result["reply"]
    assert not agent.defects(task, result)
    assert [call[1] for call in node.calls] == ["list_solutions", "list_methodologies"]


def test_an_old_engine_reports_that_only_m1_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(agent.catalog, "as_context", lambda: "")

    class OldNode(FakeNode):
        def read(self, capability, **params):
            self.calls.append(("read", capability, params))
            request = httpx.Request("GET", "https://workspace/v1/list_methodologies")
            response = httpx.Response(
                404,
                json={"error": {"code": "unknown_capability"}},
                request=request,
            )
            raise httpx.HTTPStatusError("unknown capability", request=request, response=response)

    node = OldNode({})
    client = Scripted(completion(content=json.dumps({
        "pipeline": None,
        "declined": {"methodology": "M-3",
                     "issue": "workspace engine predates the methodology registry",
                     "reason": "this older workspace can run only M-1"},
        "reply": "This engine predates the registry and only M-1 runs; M-3 cannot run.",
    })))
    task = {"project": "customer-project", "profile": {}, "intent": "segment customers"}

    result = agent.build(task, Corpus(tmp_path / "missing.jsonl"), client, "model", node, 0)

    assert result["methodology_registry"]["old_engine"] is True
    assert [item["id"] for item in result["methodology_registry"]["methodologies"]] == ["M-1"]
    assert not agent.defects(task, result)


def test_edit_mode_recompiles_churn_into_segmentation_shape():
    registry = {"methodologies": [
        method("M-1", "Churn", "gbdt"),
        method("M-3", "Segmentation / clustering", "kmeans"),
    ]}
    node = FakeNode(registry)
    answer = {
        "methodology": "M-3",
        "model_kind": "kmeans",
        "summary": "Replace churn training with customer segmentation",
        "edits": [
            {"op": "remove_step", "step": "load_activity"},
            {"op": "remove_step", "step": "build_features"},
            {"op": "remove_step", "step": "train"},
            {"op": "add_step", "kind": "clean", "name": "Prepare cluster features",
             "id": "prepare_segments"},
            {"op": "add_step", "kind": "model", "name": "Cluster customers",
             "id": "segment_customers"},
            {"op": "set_attribute", "step": "segment_customers", "name": "model_kind",
             "value": "kmeans"},
            {"op": "add_edge", "from": "load_customers", "to": "prepare_segments"},
            {"op": "add_edge", "from": "prepare_segments", "to": "segment_customers"},
        ],
        "run": {"dataset": "datasets.customers", "holdout": 0.0},
    }
    client = Scripted(
        completion(tool_calls=[edit_call("analyze_dataset", {
            "dataset": "datasets.customers"})]),
        completion(tool_calls=[edit_call("plan_features", {
            "dataset": "datasets.customers", "methodology": "M-3"})]),
        completion(tool_calls=[edit_call("compile_intent", {
            "project": "ux-review-churn", "intent": "make this a segmentation pipeline",
            "methodology": "M-3"})]),
        completion(content=json.dumps(answer)),
    )

    result = edit.propose(
        turn("make this a segmentation pipeline"), client, "model", node,
    )

    assert result["kind"] == "proposal"
    assert result["proposal"]["methodology"] == "M-3"
    assert not result["proposal"]["problems"]
    model_step = next(node for node in result["proposal"]["after"]["nodes"]
                      if node["id"] == "segment_customers")
    assert model_step["attributes"]["model_kind"] == "kmeans"
    assert any(call[0] == "act" and call[1] == "compile_intent" for call in node.calls)


def test_edit_mode_declines_segmentation_when_it_is_not_runnable():
    node = FakeNode({"methodologies": [
        method("M-3", "Segmentation / clustering", "kmeans", False, "zetta#1042")
    ]})
    client = Scripted(completion(content=json.dumps({
        "declined": {"methodology": "M-3", "issue": "zetta#1042",
                     "reason": "segmentation must become runnable"},
    })))

    result = edit.propose(turn("make this a segmentation pipeline"), client, "model", node)

    assert result["kind"] == "declined"
    assert result["declined"]["issue"] == "zetta#1042"
    assert "zetta#1042" in result["reply"]
    assert result["proposal"] is None
    assert PIPELINE["nodes"][-1]["id"] == "train"
