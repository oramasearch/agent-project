# SPDX-License-Identifier: MIT

"""Service mode: the Project Agent as a long-running agent lake calls (lake#228, agent-project#1).

The CLI builds one Pipeline from a task file and exits. In lake the agent is a specialized service
of the Agent Library: Design's *Create a Solution* and Ask Orama (as a subagent) send it turns.
One turn is one `build`, against the customer's workspace reached through lake's Zetta-compatible
path (`ZETTA_BASE_URL` points there, `ZETTA_TOKEN` is the lake key the turn carries), so the
observations are real and lake's allow-lists and audit apply. The model is reached through lake's
model gateway with the same key (`model.py`, lake#248).

Standard library only: the image carries exactly the agent's dependencies and nothing a web
framework would add.
"""

from __future__ import annotations

import hmac
import json
import os
from importlib.metadata import version
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from openai import OpenAI

from agent_project import apps, experiments, foundation, lake_bus, solutions
from agent_project.agent import build, defects
from agent_project.edit import propose, wants_edit
from agent_project.model import gateway_client
from agent_project.node import Node
from agent_project.retrieval import Corpus


class Settings:
    """What the service reads from its environment, once."""

    def __init__(self) -> None:
        self.corpus_path = Path(os.environ.get("PROJECT_CORPUS", "/app/corpus/project-sft-v1.jsonl"))
        self.model = os.environ.get("PROJECT_MODEL", "gpt-4o")
        self.token = os.environ.get("AGENT_SERVICE_TOKEN", "")
        self.trace_dir = Path(os.environ.get("TRACE_DIR", "/tmp/agent-project-traces"))
        self.default_base_url = os.environ.get("ZETTA_BASE_URL", "")
        self.shots = int(os.environ.get("PROJECT_SHOTS", "2"))


DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(url: str) -> tuple[str, str, int] | None:
    """`(scheme, host, port)` of an absolute http(s) URL with no user info, or None."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname or "@" in parts.netloc:
        return None
    return scheme, parts.hostname.lower(), port or DEFAULT_PORTS[scheme]


def lakes_url() -> str:
    """Lake's address from this service's environment (`LAKE_URL`), the only place a turn's
    credential is ever sent; empty when unset."""
    return os.environ.get("LAKE_URL", "").strip().rstrip("/")


def on_lake(requested: str, lake: str) -> str | None:
    """`requested` when it is an address on Lake at `LAKE_URL`, else None.

    Lake hands the agent its read path to the person's workspace
    (`<LAKE_URL>/w/<workspace>/workspaces/<customer>/zetta`) and its own address, and the turn's
    credential is sent there. Both come from the request, so they are accepted only on Lake's own
    origin (scheme, host and port of `LAKE_URL`) under `LAKE_URL`'s path, with no user info, query
    or fragment; anything else would send the bearer to a host the caller chose (#30).
    """
    want = _origin(lake)
    got = _origin(requested)
    if want is None or got is None or got != want:
        return None
    parts, base = urlsplit(requested), urlsplit(lake)
    if parts.query or parts.fragment or "\\" in requested:
        return None
    prefix = base.path.rstrip("/")
    path = parts.path
    if prefix and not (path == prefix or path.startswith(prefix + "/")):
        return None
    if any(segment in (".", "..") for segment in path.split("/")):
        return None
    return requested.rstrip("/")


NO_LAKE_URL = ("LAKE_URL is not set: this agent sends a turn's credential only to Lake's "
               "configured address")


def task_from_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """A turn is either a full task (the CLI's shape) or a plain-language request.

    A plain request becomes a task whose intent is the request. The profile names only what the
    caller handed over; everything else the agent must find out by calling tools, which is the
    property that makes its trajectory real (see `agent._situation`).
    """
    if isinstance(turn.get("task"), dict):
        return turn["task"]
    context = turn.get("context") or {}
    return {"profile": context.get("profile") or {"industry": context.get("industry"),
                                                  "persona": context.get("persona"),
                                                  "tables": context.get("tables") or []},
            "project": context.get("project"),
            "intent": turn.get("request") or turn.get("message") or "",
            "methodology": turn.get("methodology"),
            "solution": turn.get("solution") or context.get("solution"),
            "level": turn.get("level") or context.get("level"),
            "sources": turn.get("sources") or context.get("sources"),
            "approval": turn.get("approval") or context.get("approval"),
            "results": turn.get("results") or context.get("results"),
            "result_tables": turn.get("result_tables") or context.get("result_tables"),
            "model": context.get("model"),
            "before": turn.get("before")}


def run_turn(turn: dict[str, Any], settings: Settings, corpus: Corpus,
             client: OpenAI | None = None, node: Node | None = None,
             lake: apps.LakeReader | None = None) -> dict[str, Any]:
    """One build, recorded as a trace. A failure is returned, never raised past the handler."""
    task = task_from_turn(turn)
    editing = wants_edit(turn)
    proposing_foundation = str(turn.get("kind") or "").startswith("solution:")
    if editing and not (turn.get("request") or turn.get("message")):
        return {"ok": False, "error": "a change request needs a request"}
    if not editing and not task.get("intent"):
        return {"ok": False, "error": "a turn needs a request (or a task with an intent)"}
    # A turn's credential goes only to Lake at this service's own LAKE_URL, never to an address
    # the request names (#30). Every address is settled here, before any request goes out.
    lake_address = lakes_url()
    owned_node = node is None
    if node is None:
        requested = turn.get("zetta_base_url")
        token = turn.get("zetta_token") or os.environ.get("ZETTA_TOKEN", "")
        if not (requested or settings.default_base_url) or not token:
            return {"ok": False, "error": "no workspace to build against: the turn needs "
                                          "zetta_base_url and zetta_token"}
        if requested:
            if not lake_address or _origin(lake_address) is None:
                return {"ok": False, "error": NO_LAKE_URL}
            base_url = on_lake(str(requested), lake_address)
            if base_url is None:
                return {"ok": False, "error": "zetta_base_url is not Lake's workspace route at "
                                              "LAKE_URL; the turn's credential is never sent "
                                              "elsewhere"}
        else:
            # The operator's own ZETTA_BASE_URL, not an address from the request.
            base_url = settings.default_base_url
    needs_model = client is None and not proposing_foundation
    needs_lake = lake is None and (apps.handoff_inputs(turn) is not None
                                   or turn.get("lake_url") is not None)
    lake_base = lake_address
    if (needs_model or needs_lake) and (not lake_address or _origin(lake_address) is None):
        return {"ok": False, "error": NO_LAKE_URL}
    if needs_lake and turn.get("lake_url"):
        lake_base = on_lake(str(turn["lake_url"]), lake_address) or ""
        if not lake_base:
            return {"ok": False, "error": "lake_url is not on Lake at LAKE_URL; the turn's "
                                          "credential is never sent elsewhere"}
    if needs_model:
        # The model is reached through lake with the turn's own key, which counts its tokens in
        # the turn's workspace (lake#248); this agent holds no provider key. The address is this
        # service's LAKE_URL; the turn's own `lake_url` is never used for it (#30).
        model_key = turn.get("lake_token") or turn.get("zetta_token") or ""
        if not model_key:
            return {"ok": False, "error": "a turn needs its lake_token to reach the model "
                                          "through lake"}
        client = gateway_client(lake_address, model_key)
    if node is None:
        node = Node(base_url=base_url, token=token)
    try:
        if proposing_foundation:
            try:
                result = foundation.propose(turn, node)
            except Exception as error:
                return {"ok": False, "error": f"{type(error).__name__}: {error}"}
        else:
            solution_registry = solutions.read(node)
        if proposing_foundation:
            result["defects"] = []
            result["applied"] = False
        elif editing:
            # agent-project#4: a change is a proposal lake applies on the person's confirmation.
            result = propose(turn, client, turn.get("model") or settings.model, node)
            result["solution_registry"] = solution_registry.public()
            result["reads"].insert(0, {
                "call": "list_solutions",
                "args": {},
                "observation": solution_registry.observation(),
            })
        else:
            # A Lake turn's key grants no solution action (lake#796): a named solution is
            # proposed for the person to confirm, never built by the turn.
            result = build(task, corpus, client, turn.get("model") or settings.model,
                           node, settings.shots, solution_registry, act=False)
    finally:
        if owned_node:
            node.close()
    if not editing and not proposing_foundation:
        result["defects"] = defects(task, result)
        result["applied"] = False
    app_handoff = apps.handoff_inputs(turn)
    experiment_handoff = experiments.handoff_inputs(turn, result)
    if app_handoff is not None or experiment_handoff is not None:
        owned_lake = lake is None
        if lake is None:
            lake_token = turn.get("lake_token")
            if not lake_base or not lake_token:
                kind = "an App" if app_handoff is not None else "an Experiments"
                return {
                    "ok": False,
                    "error": f"{kind} handoff needs LAKE_URL and lake_token",
                }
            # LAKE_URL, or the turn's lake_url only when it is on Lake at LAKE_URL (#30).
            lake = apps.Lake(lake_base, str(lake_token))
        try:
            if experiment_handoff is not None:
                result = experiments.attach(turn, result, lake)
            if app_handoff is not None:
                result = apps.attach_handoff(turn, result, lake)
        finally:
            if owned_lake:
                lake.close()
    trace_id = str(uuid.uuid4())
    settings.trace_dir.mkdir(parents=True, exist_ok=True)
    (settings.trace_dir / f"{trace_id}.json").write_text(json.dumps(
        {"trace_id": trace_id, "agent": "agent-project",
         "at": datetime.now(timezone.utc).isoformat(),
         "turn": {**turn, "lake_token": None, "zetta_token": None},
         "task": task, "result": result}, indent=1, default=str))
    return {"ok": True, "trace_id": trace_id, "result": result}


def health(settings: Settings, corpus: Corpus) -> tuple[int, dict[str, Any]]:
    """Unhealthy without the corpus: answering zero-shot is a different agent, and a service
    that silently became one would be worse than one that says it cannot start."""
    if not corpus.examples:
        return 503, {"ok": False, "error": f"no corpus at {settings.corpus_path}",
                     "version": version("agent-project")}
    return 200, {"ok": True, "agent": "agent-project", "version": version("agent-project"),
                 "examples": len(corpus.examples), "model": settings.model}


def turn_refusal(authorization: str | None, token: str) -> tuple[int, str] | None:
    """Why `POST /v1/turn` is refused, as `(status, error)`, or None when the caller may proceed.

    The route fails closed: with `AGENT_SERVICE_TOKEN` unset or empty every turn is refused, so a
    service started without it is never open. The bearer is compared in constant time
    (`hmac.compare_digest` over bytes), so the comparison's timing says nothing about the token.
    """
    if not token:
        return 503, "AGENT_SERVICE_TOKEN is not set; this service refuses every turn"
    expected = f"Bearer {token}".encode()
    supplied = (authorization or "").encode("utf-8", "surrogateescape")
    if not hmac.compare_digest(supplied, expected):
        return 401, "unauthorized"
    return None


def make_handler(settings: Settings, corpus: Corpus) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 (the stdlib's name)
            if self.path == "/healthz":
                self._send(*health(settings, corpus))
                return
            self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/turn":
                self._send(404, {"ok": False, "error": "not found"})
                return
            refusal = turn_refusal(self.headers.get("Authorization"), settings.token)
            if refusal is not None:
                status, error = refusal
                self._send(status, {"ok": False, "error": error})
                return
            try:
                turn = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            except (json.JSONDecodeError, ValueError):
                self._send(400, {"ok": False, "error": "the body is not JSON"})
                return
            try:
                answer = run_turn(turn if isinstance(turn, dict) else {}, settings, corpus)
            except Exception as error:  # a failed build is an answer, not a dead worker
                answer = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            self._send(200 if answer.get("ok") else 422, answer)

        def log_message(self, format: str, *args: Any) -> None:
            print(json.dumps({"agent": "agent-project", "request": format % args}), flush=True)

    return Handler


def serve() -> int:
    settings = Settings()
    corpus = Corpus(settings.corpus_path)
    port = int(os.environ.get("PORT", "8080"))
    print(json.dumps({"agent": "agent-project", "listening": port,
                      "examples": len(corpus.examples)}), flush=True)
    # Turns arrive as tasks on lake's agent bus (lake#269); /v1/turn stays for direct calls.
    lake_bus.start_from_env(lambda turn: run_turn(turn, settings, corpus), "agent-project")
    ThreadingHTTPServer(("0.0.0.0", port), make_handler(settings, corpus)).serve_forever()
    return 0
