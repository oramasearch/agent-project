"""Task credentials by reference (lake#871, lake docs/api/agent-bus.md).

A claim asks Lake for `credentials: by_reference`: the body arrives with no credential, and every
call made for the task carries this agent's bus credential and `X-Lake-Agent-Task: <task id>`. A
Lake from before lake#871 refuses the field; the claim is asked again and the body's own credential
is used, as before.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from pathlib import Path
from typing import Any

import httpx

from agent_project import lake_bus
from test_turn_destinations import APP_CONTEXT, LAKE, run, sent, turn  # noqa: F401 (fixture)

TASK = {"task_id": "t1", "customer_id": "cust-1", "workspace_id": "ws-1", "kind": "api"}


class BusLake:
    """lake's bus in memory. `supports_by_reference` False is a Lake from before lake#871."""

    def __init__(self, supports_by_reference: bool) -> None:
        self.supports_by_reference = supports_by_reference
        self.claims: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.stop = threading.Event()
        self.received = False

    def send(self, method: str, url: str, headers: dict[str, str],
             body: bytes | None) -> tuple[int, str]:
        self.headers.append(headers)
        capability = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
        if capability == "receive_agent_tasks":
            tasks = [] if self.received else [{"sequence": 1, **TASK}]
            self.received = True
            if not tasks:
                self.stop.set()
            return 200, json.dumps({"ok": True, "value": {"tasks": tasks, "next_sequence": 2}})
        arguments = json.loads(body or b"{}")
        if capability == "claim_agent_task":
            self.claims.append(arguments)
            if "credentials" in arguments and not self.supports_by_reference:
                return 400, json.dumps({"ok": False, "error": {
                    "code": "invalid_input", "message": "unexpected field credentials",
                    "path": "/credentials"}})
            by_reference = arguments.get("credentials") == "by_reference"
            credential = "" if by_reference else "lat.t1.secret"
            value = {**TASK, "body": {"request": "hi", "lake_token": credential,
                                      "zetta_token": credential}}
            if by_reference:
                value["credentials"] = "by_reference"
            return 200, json.dumps({"ok": True, "value": value})
        return 200, json.dumps({"ok": True, "value": {}})


def handed_body(lake: BusLake) -> dict[str, Any]:
    handed: list[dict[str, Any]] = []
    bus = lake_bus.LakeBus("http://lake", "bus-token", "agent-project", send=lake.send)
    lake_bus.work(bus, lambda body: handed.append(body) or {"ok": True}, lake.stop, workers=1)
    assert len(handed) == 1, handed
    return handed[0]


def test_a_claim_asks_by_reference_and_the_turn_holds_no_task_credential():
    lake = BusLake(supports_by_reference=True)
    body = handed_body(lake)
    assert lake.claims == [{"task_id": "t1", "agent": "agent-project", "credentials": "by_reference"}]
    assert body["lake_token"] == body["zetta_token"] == "bus-token", \
        "the turn calls Lake with the bus credential"
    assert body["lake_headers"] == {"X-Lake-Agent-Task": "t1"}
    assert lake_bus.task_headers(body) == {"X-Lake-Agent-Task": "t1"}
    # Bus calls themselves never name a task: Lake would answer them as the task.
    assert all(lake_bus.TASK_HEADER not in headers for headers in lake.headers)


def test_a_lake_from_before_by_reference_is_claimed_again_with_the_credential_in_the_body():
    lake = BusLake(supports_by_reference=False)
    body = handed_body(lake)
    assert lake.claims == [
        {"task_id": "t1", "agent": "agent-project", "credentials": "by_reference"},
        {"task_id": "t1", "agent": "agent-project"},
    ]
    assert body["lake_token"] == "lat.t1.secret", "the body's own credential is used, as before"
    assert "lake_headers" not in body
    assert lake_bus.task_headers(body) == {}


def test_a_claim_refused_for_another_reason_is_not_asked_again():
    assert not lake_bus.refuses_by_reference(
        lake_bus.BusError("invalid_input", "task_id is required"))
    assert not lake_bus.refuses_by_reference(lake_bus.BusError("forbidden", "credentials"))
    assert lake_bus.refuses_by_reference(
        lake_bus.BusError("invalid_input", "unexpected field credentials"))


def test_task_headers_take_only_the_task_header():
    assert lake_bus.task_headers({"lake_headers": {"X-Lake-Agent-Task": "t1",
                                                   "Authorization": "Bearer other"}}) == {
        "X-Lake-Agent-Task": "t1"}
    assert lake_bus.task_headers({"lake_headers": "t1"}) == {}
    assert lake_bus.task_headers({}) == {}


def by_reference_turn(**extra):
    """A turn as `lake_bus` hands it after a claim by reference."""
    claimed = {**TASK, "credentials": "by_reference",
               "body": turn(zetta_token="", lake_token="", **extra)}
    body = lake_bus.task_credentials("bus-token", claimed)
    assert body["zetta_base_url"] and body["lake_url"]
    return body


def test_every_call_for_a_by_reference_turn_names_its_task(tmp_path, monkeypatch, sent):
    """The workspace Node, the App handoff's Lake reads and the model calls all carry the bus
    credential and `X-Lake-Agent-Task`."""
    monkeypatch.setenv("LAKE_URL", LAKE)
    answer = run(tmp_path, by_reference_turn(context=APP_CONTEXT, customer_id="cust-1"))
    assert answer["ok"], answer
    paths = [request.url.path for request in sent]
    assert any(path.startswith("/w/ws-1/workspaces/cust-1/zetta/") for path in paths)
    assert "/openai/v1/chat/completions" in paths
    assert any(path.startswith("/v1/") for path in paths), "the App handoff never read Lake"
    for request in sent:
        assert request.headers["authorization"] == "Bearer bus-token"
        assert request.headers["x-lake-agent-task"] == "t1"


def test_the_trace_keeps_no_credential_of_a_by_reference_turn(tmp_path, monkeypatch, sent):
    monkeypatch.setenv("LAKE_URL", LAKE)
    answer = run(tmp_path, by_reference_turn())
    assert answer["ok"], answer
    trace = (tmp_path / "traces" / f"{answer['trace_id']}.json").read_text()
    assert "bus-token" not in trace


def test_an_in_body_turn_sends_no_task_header(tmp_path, monkeypatch, sent):
    monkeypatch.setenv("LAKE_URL", LAKE)
    answer = run(tmp_path, turn(context=APP_CONTEXT, customer_id="cust-1"))
    assert answer["ok"], answer
    assert sent
    assert all("x-lake-agent-task" not in request.headers for request in sent)


def test_the_environments_zetta_token_never_carries_the_task_header(monkeypatch):
    """A turn without its own zetta_token uses ZETTA_TOKEN, which is not made for the task."""
    from agent_project.node import Node

    monkeypatch.setenv("ZETTA_TOKEN", "env-token")
    node = Node(base_url="http://zetta", token=None, headers={"X-Lake-Agent-Task": "t1"})
    try:
        assert node.client.headers["authorization"] == "Bearer env-token"
        assert "x-lake-agent-task" not in node.client.headers
    finally:
        node.close()
