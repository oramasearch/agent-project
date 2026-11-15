"""A turn's credential goes only to Lake at LAKE_URL, never to an address the request names (#30)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from agent_project import apps, service
from agent_project import node as node_module
from agent_project.retrieval import Corpus

LAKE = "http://lake-prod.lake.svc.cluster.local:80"
ROUTE = LAKE + "/w/ws-1/workspaces/cust-1/zetta"
APP_CONTEXT = {"project": "retention", "results": {
    "kind": "scored-list", "tables": ["results.account_risk"],
    "model": {"graph": "churn", "step": "score_accounts"}}}


def settings(tmp_path: Path) -> service.Settings:
    value = service.Settings()
    value.trace_dir = tmp_path / "traces"
    value.default_base_url = ""
    return value


@pytest.fixture
def sent(monkeypatch):
    """Every request a workspace Node, a Lake reader or the model client makes."""
    requests: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "value": {"solutions": []}})

    real = httpx.Client

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(answer)
        return real(*args, **kwargs)

    monkeypatch.setattr(node_module.httpx, "Client", client)
    monkeypatch.setattr(apps.httpx, "Client", client)

    def gateway(url, token, headers=None):
        requests.append(httpx.Request("POST", f"{url}/openai/v1/chat/completions",
                                      headers={**(headers or {}),
                                               "authorization": f"Bearer {token}"}))
        return object()

    monkeypatch.setattr(service, "gateway_client", gateway)
    monkeypatch.setattr(service, "build", lambda *_a, **_k: {"pipeline": {}, "reply": ""})
    monkeypatch.setattr(service, "defects", lambda *_a, **_k: [])
    return requests


def turn(**extra):
    return {"request": "predict churn", "zetta_base_url": ROUTE, "zetta_token": "lat.task.secret",
            "lake_url": LAKE, "lake_token": "lat.task.secret", **extra}


def run(tmp_path, value):
    return service.run_turn(value, settings(tmp_path), Corpus(tmp_path / "missing.jsonl"))


OFF_LAKE = [
    "http://attacker.example/w/ws-1/workspaces/cust-1/zetta",
    "https://lake-prod.lake.svc.cluster.local/w/ws-1/workspaces/cust-1/zetta",
    "http://lake-prod.lake.svc.cluster.local:8080/w/ws-1/workspaces/cust-1/zetta",
    "http://lake-prod.lake.svc.cluster.local.attacker.example/w/ws-1/zetta",
    "http://user@lake-prod.lake.svc.cluster.local/w/ws-1/zetta",
    "http://lake-prod.lake.svc.cluster.local:80@attacker.example/zetta",
    "http://lake-prod.lake.svc.cluster.local/w/ws-1/zetta?next=http://attacker.example",
    "file:///etc/passwd",
    "lake-prod.lake.svc.cluster.local/w/ws-1/zetta",
    "http://lake-prod.lake.svc.cluster.local:notaport/zetta",
]


@pytest.mark.parametrize("zetta_base_url", OFF_LAKE)
def test_a_workspace_address_off_lake_is_refused_before_any_request(
        tmp_path, monkeypatch, sent, zetta_base_url):
    monkeypatch.setenv("LAKE_URL", LAKE)
    answer = run(tmp_path, turn(zetta_base_url=zetta_base_url))
    assert not answer["ok"]
    assert "zetta_base_url is not Lake's workspace route" in answer["error"]
    assert sent == []


@pytest.mark.parametrize("lake_url", OFF_LAKE)
def test_a_lake_address_off_lake_is_refused_before_any_request(
        tmp_path, monkeypatch, sent, lake_url):
    monkeypatch.setenv("LAKE_URL", LAKE)
    answer = run(tmp_path, turn(lake_url=lake_url, context=APP_CONTEXT))
    assert not answer["ok"]
    assert "lake_url is not on Lake at LAKE_URL" in answer["error"]
    assert sent == []


def test_every_request_goes_to_lake_at_lake_url_with_the_turns_credential(
        tmp_path, monkeypatch, sent):
    monkeypatch.setenv("LAKE_URL", LAKE)
    answer = run(tmp_path, turn(context=APP_CONTEXT, customer_id="cust-1"))
    assert answer["ok"], answer
    paths = [request.url.path for request in sent]
    assert any(path.startswith("/w/ws-1/workspaces/cust-1/zetta/") for path in paths)
    assert "/openai/v1/chat/completions" in paths
    assert any(path.startswith("/v1/") for path in paths), "the App handoff never read Lake"
    assert all(request.url.host == "lake-prod.lake.svc.cluster.local"
               and request.url.port in (None, 80) for request in sent)
    assert all(request.headers["authorization"] == "Bearer lat.task.secret" for request in sent)


def test_the_model_ignores_the_turns_lake_url(tmp_path, monkeypatch, sent):
    monkeypatch.setenv("LAKE_URL", LAKE)
    answer = run(tmp_path, turn(lake_url=ROUTE))
    assert answer["ok"], answer
    model = [r for r in sent if r.url.path.endswith("/chat/completions")]
    assert [(r.url.host, r.url.path) for r in model] == [
        ("lake-prod.lake.svc.cluster.local", "/openai/v1/chat/completions")]


def test_without_lake_url_no_turn_sends_its_credential(tmp_path, monkeypatch, sent):
    monkeypatch.delenv("LAKE_URL", raising=False)
    answer = run(tmp_path, turn())
    assert not answer["ok"] and "LAKE_URL is not set" in answer["error"]
    assert sent == []
    answer = run(tmp_path, turn(kind="solution:propose"))
    assert not answer["ok"] and "LAKE_URL is not set" in answer["error"]
    assert sent == []


def test_an_address_must_be_under_lake_urls_path():
    lake = "http://lake:80/base"
    assert service.on_lake("http://lake/base/w/1/zetta", lake) == "http://lake/base/w/1/zetta"
    assert service.on_lake("http://LAKE/base", lake) == "http://LAKE/base"
    assert service.on_lake("http://lake/basement/w/1/zetta", lake) is None
    assert service.on_lake("http://lake/other/w/1/zetta", lake) is None
    assert service.on_lake("http://lake/base/../admin", lake) is None
    assert service.on_lake("http://lake/base\\@attacker.example/", lake) is None
