"""Experiment handoffs use registered Lake reads and leave every write for confirmation."""

from __future__ import annotations

import json
from pathlib import Path

from agent_project import experiments
from agent_project import service
from agent_project.retrieval import Corpus


LAKE_MCP_TOOLS_LIST = {
    "tools": [
        {"name": name}
        for name in (
            "list_experiments",
            "get_experiment_study",
            "list_experiment_trials",
            "get_experiment_analysis",
            "get_run_metrics",
            "compare_runs",
            "tail_experiment_events",
            "start_study",
            "stop_study",
            "stop_experiment_trial",
            "promote_trial",
        )
    ],
}


class FakeLake:
    def __init__(self, studies=None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.studies = studies or []

    def read(self, capability: str, **params):
        self.calls.append((capability, params))
        if capability == "list_experiments":
            return {"studies": self.studies, "runs": []}
        if capability == "get_experiment_study":
            return {
                "id": params["study"],
                "spec": {
                    "project": "retention",
                    "methodology": "M-1",
                },
                "status": "complete",
                "trials": [{"number": 1}, {"number": 2}],
                "spent_cost": 1.5,
                "promoted_model": "model-best",
            }
        raise AssertionError(f"unexpected capability {capability}")

    def act(self, capability: str, **params):
        raise AssertionError(f"the Project Agent cannot act: {capability} {params}")


def project_result(model_kinds=None):
    choices = ["gbdt", "linear"] if model_kinds is None else model_kinds
    return {
        "pipeline": {
            "dataset": "datasets.accounts",
            "methodology": "M-1",
            "model_kind": "gbdt",
            "label": "churned",
            "features": ["tenure"],
        },
        "methodology_registry": {
            "methodologies": [{
                "id": "M-1",
                "runnable": True,
                "model_kinds": choices,
                "metrics": ["validation.auc"],
            }],
        },
        "reply": "Designed the retention pipeline.",
    }


def project_turn():
    return {
        "request": "predict churn",
        "customer_id": "customer-1",
        "context": {"project": "retention"},
    }


def test_experiment_tools_are_registered_lake_mcp_reads_and_no_write():
    names = {item["function"]["name"] for item in experiments.specs()}
    registered = {item["name"] for item in LAKE_MCP_TOOLS_LIST["tools"]}

    assert names == set(experiments.READS)
    assert names <= registered
    assert not names & experiments.PROPOSED_ONLY_ACTIONS


def test_project_studies_are_listed_and_the_matching_study_is_read():
    lake = FakeLake(studies=[{
        "id": "study-1",
        "spec": {"project": "retention", "methodology": "M-1"},
        "status": "complete",
    }])
    inputs = experiments.handoff_inputs(project_turn(), project_result())

    evidence = experiments.read_state(lake, inputs)

    assert lake.calls == [
        (
            "list_experiments",
            {"customer_id": "customer-1", "project": "retention"},
        ),
        (
            "get_experiment_study",
            {"customer_id": "customer-1", "study": "study-1"},
        ),
    ]
    assert evidence.matching_study == "study-1"
    assert experiments.study_summary(evidence) == (
        "Study study-1 is complete with 2 trial(s). Its promoted model is model-best."
    )


def test_a_new_model_selection_is_a_confirmation_only_agent_proposal():
    lake = FakeLake()

    result = experiments.attach(project_turn(), project_result(), lake)

    proposal = result["experiment_handoff"]
    assert proposal["kind"] == "proposal"
    assert proposal["capability"] == "agent_turn"
    assert proposal["agent"] == "agent-experiments"
    assert proposal["turn_kind"] == "experiment"
    assert proposal["confirmation_required"] is True
    assert proposal["request"]["context"] == {
        "project": "retention",
        "dataset": "datasets.accounts",
        "methodology": "M-1",
        "model_kinds": ["gbdt", "linear"],
        "provisional_model_kind": "gbdt",
        "label": "churned",
        "features": ["tenure"],
        "metrics": ["validation.auc"],
    }
    assert "do not start a study or promote a trial" in proposal["request"]["message"]
    assert "Proposed, not started" in result["reply"]
    assert lake.calls == [(
        "list_experiments",
        {"customer_id": "customer-1", "project": "retention"},
    )]


def test_an_existing_study_is_explained_without_a_duplicate_proposal():
    lake = FakeLake(studies=[{
        "id": "study-1",
        "spec": {"methodology": "M-1"},
        "status": "complete",
    }])

    result = experiments.attach(project_turn(), project_result(), lake)

    assert "experiment_handoff" not in result
    assert result["experiment_summary"].startswith("Study study-1 is complete")
    assert "model-best" in result["reply"]


def test_a_methodology_without_a_model_does_not_offer_a_study():
    assert experiments.handoff_inputs(
        project_turn(),
        project_result(model_kinds=[]),
    ) is None


def test_one_model_kind_still_offers_hyperparameter_tuning():
    result = experiments.attach(
        project_turn(),
        project_result(model_kinds=["gbdt"]),
        FakeLake(),
    )

    assert result["experiment_handoff"]["inputs"]["model_kinds"] == ["gbdt"]
    assert "tune consequential hyperparameters" in (
        result["experiment_handoff"]["request"]["message"]
    )


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


def test_service_attaches_the_study_offer_and_scrubs_both_credentials(
        tmp_path,
        monkeypatch):
    built = project_result()
    monkeypatch.setattr(service, "build", lambda *_args, **_kwargs: built)
    monkeypatch.setattr(service, "defects", lambda *_args, **_kwargs: [])
    monkeypatch.setenv("LAKE_URL", "http://lake")
    lake = FakeLake()
    turn = {
        **project_turn(),
        "lake_url": "http://lake",
        "lake_token": "lat.task.secret",
        "zetta_token": "lat.task.secret",
    }

    answer = service.run_turn(
        turn,
        settings(tmp_path),
        Corpus(tmp_path / "missing.jsonl"),
        node=ProjectNode(),
        lake=lake,
    )

    assert answer["ok"] is True
    assert answer["result"]["experiment_handoff"]["agent"] == "agent-experiments"
    trace = json.loads(
        (tmp_path / "traces" / f"{answer['trace_id']}.json").read_text()
    )
    assert trace["turn"]["lake_token"] is None
    assert trace["turn"]["zetta_token"] is None
