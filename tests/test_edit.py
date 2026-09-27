"""Edit mode: a change request answers with a checked proposal and never claims it was applied."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent_project import edit, service
from agent_project.retrieval import Corpus

PIPELINE = {
    "name": "churn skeleton",
    "nodes": [
        {"id": "load_customers", "kind": "connect", "name": "Load customer accounts",
         "attributes": {"source": "warehouse", "table": "customers"}},
        {"id": "load_activity", "kind": "connect", "name": "Load usage & billing activity",
         "attributes": {"source": "warehouse", "table": "events"}},
        {"id": "build_features", "kind": "clean", "name": "Build churn features", "attributes": {}},
        {"id": "train", "kind": "model", "name": "Train", "attributes": {"target": "churned"}},
    ],
    "edges": [
        {"from": "load_customers", "to": "build_features", "kind": "data_flow"},
        {"from": "load_activity", "to": "build_features", "kind": "data_flow"},
        {"from": "build_features", "to": "train", "kind": "data_flow"},
    ],
}


def completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    message.model_dump = lambda exclude_none=True: {"role": "assistant", "content": content or ""}
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def call(name, args):
    return SimpleNamespace(id=f"call-{name}",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_):
        return self.replies.pop(0)


class FakeNode:
    def __init__(self):
        self.reads = []

    def read(self, capability, **params):
        self.reads.append((capability, params))
        if capability == "list_datasets":
            return {"datasets": [{"database": "datasets", "name": "ux_review_churn"}]}
        return {"columns": ["account_id", "churned", "plan"], "rows": 200}

    def close(self):
        pass


def turn(request="Drop the activity step, train on ux_review_churn, hold out 20%"):
    return {"request": request, "context": {"project": "ux-review-churn", "graph": "churn skeleton",
                                            "pipeline": PIPELINE}}


ANSWER = {"summary": "Train on ux_review_churn only, with a 20% hold-out",
          "edits": [{"op": "remove_step", "step": "load_activity"},
                    {"op": "set_attribute", "step": "load_customers", "name": "source",
                     "value": "datasets"},
                    {"op": "set_attribute", "step": "load_customers", "name": "table",
                     "value": "ux_review_churn"}],
          "run": {"dataset": "datasets.ux_review_churn", "holdout": 0.2}}


def test_a_change_request_is_recognized_by_its_pipeline_context():
    assert edit.wants_edit(turn())
    assert not edit.wants_edit({"request": "build me a churn pipeline"})


def test_simulate_applies_edits_in_order_like_lake():
    after, problems = edit.simulate(PIPELINE, [dict(e) for e in ANSWER["edits"]])
    assert not problems
    assert [n["id"] for n in after["nodes"]] == ["load_customers", "build_features", "train"]
    assert all("load_activity" not in (e["from"], e["to"]) for e in after["edges"])
    assert after["nodes"][0]["attributes"] == {"source": "datasets", "table": "ux_review_churn"}
    assert PIPELINE["nodes"][1]["id"] == "load_activity", "the original graph is untouched"


def test_simulate_names_every_edit_that_would_not_apply():
    _, problems = edit.simulate(PIPELINE, [
        {"op": "remove_step", "step": "nope"},
        {"op": "add_edge", "from": "train", "to": "train"},
        {"op": "add_step", "kind": "teleport", "name": "x"},
        {"op": "explode"},
        {"op": "rename_step", "step": "train"}])
    assert len(problems) == 5


def test_a_proposal_reads_the_data_and_never_claims_it_was_applied():
    # The model insists it changed the pipeline; the reply is written from the edits, not by it.
    claimed = {**ANSWER, "summary": "I updated the pipeline in place"}
    client = Scripted(completion(tool_calls=[call("list_datasets", {})]),
                      completion(content="```json\n" + json.dumps(claimed) + "\n```"))
    node = FakeNode()
    result = edit.propose(turn(), client, "gpt-5.5", node)
    assert node.reads and node.reads[0][0] == "list_datasets"
    assert result["kind"] == "proposal"
    assert result["reply"].startswith("Proposed, not applied")
    proposal = result["proposal"]
    assert proposal["project"] == "ux-review-churn" and proposal["graph"] == "churn skeleton"
    assert proposal["run"] == {"dataset": "datasets.ux_review_churn", "holdout": 0.2}
    assert proposal["steps"][0] == "Remove step load_activity and its edges"
    assert not proposal["problems"]


def test_an_edit_that_would_not_apply_is_reported_not_proposed_as_fine():
    bad = {**ANSWER, "edits": [{"op": "remove_step", "step": "load_everything"}],
           "run": {"holdout": 3}}
    result = edit.propose(turn(), Scripted(completion(content=json.dumps(bad))), "m", FakeNode())
    assert "would not apply" in result["reply"] and "Nothing was changed" in result["reply"]
    assert len(result["proposal"]["problems"]) == 2


def test_the_service_routes_a_change_request_to_a_proposal(tmp_path: Path):
    settings = service.Settings()
    settings.trace_dir = tmp_path
    client = Scripted(completion(content=json.dumps(ANSWER)))
    answer = service.run_turn(turn(), settings, Corpus(tmp_path / "none.jsonl"), client=client,
                              node=FakeNode())
    assert answer["ok"] and answer["result"]["kind"] == "proposal"
    assert answer["result"]["reply"].startswith("Proposed, not applied")
