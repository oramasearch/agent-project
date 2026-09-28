"""Service mode without a model or a node: the turn contract, the health rule, and the trace."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent_project import service
from agent_project.retrieval import Corpus


class FakeNode:
    """Answers the reads the way a Zetta node does, recording what was asked."""

    def __init__(self) -> None:
        self.reads: list[tuple[str, dict]] = []

    def read(self, capability: str, **params):
        self.reads.append((capability, params))
        if capability == "list_methodologies":
            return {"methodologies": [{
                "id": "M-1", "name": "Churn", "model_kinds": ["gbdt"],
                "inputs": [{"role": "label", "required": True}], "runnable": True,
            }]}
        if capability == "list_datasets":
            return {"datasets": [{"database": "datasets", "name": "accounts"}]}
        if capability == "analyze_dataset":
            return {"grain": "snapshot", "keys": ["account_id"]}
        if capability == "plan_features":
            return {"features": ["tenure"]}
        return {"columns": [{"name": "churned", "type": "bool", "null_fraction": 0.0}]}

    def act(self, capability: str, **params):
        assert capability == "compile_intent"
        return {"methodology": params["methodology"], "nodes": ["connect", "model"]}

    def close(self) -> None:
        pass


def completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    message.model_dump = lambda exclude_none=True: {"role": "assistant", "content": content or ""}
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def call(name, args, because=None):
    payload = {"args": args}
    if because:
        payload["because"] = because
    return SimpleNamespace(id=f"call-{name}",
                           function=SimpleNamespace(name=name, arguments=json.dumps(payload)))


class ScriptedClient:
    """A model that reads, then builds, then answers: the loop's shape with no network."""

    def __init__(self) -> None:
        self.replies = [
            completion(tool_calls=[call("list_datasets", {},
                                        because="list_methodologies returned runnable M-1")]),
            completion(tool_calls=[call("get_dataset", {"dataset": "datasets.accounts"},
                                        because="list_datasets returned accounts")]),
            completion(tool_calls=[call("analyze_dataset", {"dataset": "datasets.accounts"},
                                        because="get_dataset returned churned")]),
            completion(tool_calls=[call("plan_features", {
                "dataset": "datasets.accounts", "methodology": "M-1"},
                because="analyze_dataset returned snapshot grain")]),
            completion(tool_calls=[call("compile_intent", {
                "project": "churn", "intent": "predict churn", "methodology": "M-1"},
                because="plan_features returned tenure")]),
            completion(content='```json\n{"pipeline": {"dataset": "datasets.accounts", '
                               '"label": "churned", "methodology": "M-1", '
                               '"model_kind": "gbdt", "features": ["tenure"]}, '
                               '"reply": "Designed."}\n```'),
        ]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_):
        return self.replies.pop(0)


def settings(tmp_path: Path) -> service.Settings:
    value = service.Settings()
    value.trace_dir = tmp_path / "traces"
    value.corpus_path = tmp_path / "missing.jsonl"
    return value


def test_a_plain_request_becomes_a_task_that_must_discover_its_data():
    task = service.task_from_turn({"request": "predict churn", "context": {"industry": "saas"}})
    assert task["intent"] == "predict churn"
    assert task["profile"]["industry"] == "saas"
    assert task["profile"]["tables"] == []


def test_a_turn_builds_against_the_node_and_records_a_trace(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_project.agent.catalog.as_context", lambda: "")
    node = FakeNode()
    answer = service.run_turn({"request": "predict churn"}, settings(tmp_path),
                              Corpus(tmp_path / "missing.jsonl"), ScriptedClient(), node)
    assert answer["ok"]
    assert answer["result"]["pipeline"]["methodology"] == "M-1"
    assert [step["call"] for step in answer["result"]["trajectory"]] == [
        "list_methodologies", "list_datasets", "get_dataset", "analyze_dataset",
        "plan_features", "compile_intent",
    ]
    trace = json.loads((tmp_path / "traces" / f"{answer['trace_id']}.json").read_text())
    assert trace["agent"] == "agent-project"
    assert trace["turn"]["zetta_token"] is None, "a trace never keeps the caller's key"


def test_reads_go_to_the_nodes_real_capabilities():
    from agent_project.node import READ_CAPABILITY
    assert READ_CAPABILITY == {
        "list_solutions": "list_solutions", "get_solution": "get_solution",
        "get_policy": "get_policy", "get_experiment": "get_experiment",
        "list_ledger": "list_ledger", "get_monitor": "get_monitor",
        "get_roi_readout": "get_roi_readout",
        "list_methodologies": "list_methodologies", "list_datasets": "list_datasets",
        "get_dataset": "get_dataset", "analyze_dataset": "analyze_dataset",
        "plan_features": "plan_features", "list_pipelines": "list_projects",
        "list_explainers": "list_models",
    }


def test_a_turn_without_a_request_or_a_workspace_is_refused(tmp_path):
    corpus = Corpus(tmp_path / "missing.jsonl")
    assert not service.run_turn({}, settings(tmp_path), corpus)["ok"]
    refused = service.run_turn({"request": "x"}, settings(tmp_path), corpus)
    assert not refused["ok"] and "zetta_base_url" in refused["error"]


def test_health_fails_without_the_corpus(tmp_path):
    status, body = service.health(settings(tmp_path), Corpus(tmp_path / "missing.jsonl"))
    assert status == 503 and not body["ok"]
