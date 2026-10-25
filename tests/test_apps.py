"""App handoffs use Lake reads and leave every write for confirmation."""

from __future__ import annotations

import json
from pathlib import Path

from agent_project import apps
from agent_project import service
from agent_project.retrieval import Corpus


class FakeLake:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def read(self, capability: str, **params):
        self.calls.append((capability, params))
        values = {
            "list_apps": {"apps": [{"id": "app-1", "name": "risk-review"}]},
            "get_app": {"app": {"id": params.get("app_id"), "phase": "Running"}},
            "receive_app_events": {"events": [{"phase": "Running"}], "next_sequence": 2},
            "list_builds": {"builds": [{"id": "build-1", "phase": "succeeded"}]},
            "get_build": {"build": {"id": params.get("build_id"), "phase": "succeeded"}},
            "receive_build_events": {"events": [{"type": "done"}], "next_sequence": 2},
        }
        return values[capability]

    def act(self, capability: str, **params):
        raise AssertionError(f"the Project Agent cannot act: {capability} {params}")


def test_app_tools_are_the_six_lake_reads_and_no_write():
    names = {item["function"]["name"] for item in apps.specs()}

    assert names == set(apps.READS)
    assert names == {
        "list_apps",
        "get_app",
        "receive_app_events",
        "list_builds",
        "get_build",
        "receive_build_events",
    }
    assert not names & apps.PROPOSED_ONLY_ACTIONS


def test_selected_app_and_build_state_uses_every_read_capability():
    lake = FakeLake()

    evidence = apps.read_state(lake, {
        "customer_id": "customer-1",
        "app_id": "app-1",
        "build_id": "build-1",
    })

    assert [name for name, _ in lake.calls] == [
        "list_apps",
        "get_app",
        "receive_app_events",
        "list_builds",
        "get_build",
        "receive_build_events",
    ]
    assert all(item["ok"] for item in evidence.reads)
    assert lake.calls[2][1]["from_sequence"] == 1
    assert lake.calls[5][1]["wait_ms"] == 0


def test_handoff_passes_tables_and_model_to_application_agent_without_starting_it():
    lake = FakeLake()
    evidence = apps.read_state(lake, {"customer_id": "customer-1"})
    tables = ["results.account_risk"]
    model = {"project": "retention", "graph": "churn", "step": "score_accounts"}
    results = {"kind": "scored_list", "score": "risk", "entity": "account_id"}

    proposal = apps.application_agent_proposal(
        "retention",
        tables,
        model,
        results,
        evidence,
    )

    assert proposal["kind"] == "proposal"
    assert proposal["capability"] == "agent_turn"
    assert proposal["agent"] == "agent-application"
    assert proposal["turn_kind"] == "app"
    assert proposal["confirmation_required"] is True
    assert proposal["request"]["context"] == {
        "project": "retention",
        "tables": tables,
        "model": model,
        "results": results,
    }
    assert all(name in apps.PROPOSED_ONLY_ACTIONS for name in (
        "create_app",
        "deploy_app",
        "build_app",
        "rollback_app",
        "scale_app",
    ))
    assert [name for name, _ in lake.calls] == ["list_apps", "list_builds"]


class ProjectNode:
    def read(self, capability: str, **params):
        assert capability == "list_solutions"
        return {"error": {"code": "unknown_capability"}}

    def close(self) -> None:
        pass


def settings(tmp_path: Path) -> service.Settings:
    value = service.Settings()
    value.trace_dir = tmp_path / "traces"
    value.corpus_path = tmp_path / "missing.jsonl"
    return value


def test_service_offers_app_for_a_scored_result_and_scrubs_both_turn_keys(
        tmp_path,
        monkeypatch):
    built = {"pipeline": {"methodology": "M-1"}, "reply": "The scored list is ready."}
    monkeypatch.setattr(service, "build", lambda *_args, **_kwargs: built)
    monkeypatch.setattr(service, "defects", lambda *_args, **_kwargs: [])
    monkeypatch.setenv("LAKE_URL", "http://lake")
    lake = FakeLake()
    turn = {
        "request": "Present the scored retention results.",
        "customer_id": "customer-1",
        "lake_url": "http://lake",
        "lake_token": "lat.task.secret",
        "zetta_token": "lat.task.secret",
        "context": {
            "project": "retention",
            "results": {
                "kind": "scored-list",
                "tables": ["results.account_risk"],
                "model": {"graph": "churn", "step": "score_accounts"},
            },
        },
    }

    answer = service.run_turn(
        turn,
        settings(tmp_path),
        Corpus(tmp_path / "missing.jsonl"),
        node=ProjectNode(),
        lake=lake,
    )

    assert answer["ok"] is True
    result = answer["result"]
    assert result["app_handoff"]["agent"] == "agent-application"
    assert result["app_handoff"]["inputs"]["tables"] == ["results.account_risk"]
    assert result["app_handoff"]["inputs"]["model"] == {
        "graph": "churn",
        "step": "score_accounts",
    }
    assert result["applied"] is False
    assert "Proposed, not started" in result["reply"]
    assert [name for name, _ in lake.calls] == ["list_apps", "list_builds"]
    trace = json.loads((tmp_path / "traces" / f"{answer['trace_id']}.json").read_text())
    assert trace["turn"]["lake_token"] is None
    assert trace["turn"]["zetta_token"] is None


def test_non_ui_results_do_not_offer_an_app():
    assert apps.handoff_inputs({
        "customer_id": "customer-1",
        "context": {
            "project": "retention",
            "results": {
                "kind": "training_metrics",
                "tables": ["results.metrics"],
                "model": {"graph": "churn"},
            },
        },
    }) is None


def test_existing_app_is_passed_to_application_agent_as_a_change_target():
    lake = FakeLake()
    turn = {
        "customer_id": "customer-1",
        "context": {
            "project": "retention",
            "app_id": "app-1",
            "results": {
                "kind": "forecast",
                "tables": ["results.retention_forecast"],
                "model": {"graph": "forecast"},
            },
        },
    }

    result = apps.attach_handoff(turn, {"reply": "Forecast ready."}, lake)

    handoff = result["app_handoff"]
    assert handoff["request"]["context"]["app_id"] == "app-1"
    assert handoff["request"]["message"].startswith("Change App app-1")
    assert [name for name, _ in lake.calls] == [
        "list_apps",
        "get_app",
        "receive_app_events",
        "list_builds",
    ]
