"""App handoffs use Lake reads and leave every write for confirmation."""

from __future__ import annotations

from agent_project import apps


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
