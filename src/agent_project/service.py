"""Service mode: the Project Agent as a long-running agent lake calls (lake#228, agent-project#1).

The CLI builds one Pipeline from a task file and exits. In lake the agent is a specialized service
of the Agent Library: Design's *Create a Solution* and Ask Orama (as a subagent) send it turns.
One turn is one `build`, against the customer's workspace reached through lake's Zetta-compatible
path (`ZETTA_BASE_URL` points there, `ZETTA_TOKEN` is the lake key the turn carries), so the
observations are real and lake's allow-lists and audit apply.

Standard library only: the image carries exactly the agent's dependencies and nothing a web
framework would add.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent_project.agent import build, defects
from agent_project.edit import propose, wants_edit
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
            "intent": turn.get("request") or turn.get("message") or "",
            "methodology": turn.get("methodology"),
            "before": turn.get("before")}


def run_turn(turn: dict[str, Any], settings: Settings, corpus: Corpus,
             client: OpenAI | None = None, node: Node | None = None) -> dict[str, Any]:
    """One build, recorded as a trace. A failure is returned, never raised past the handler."""
    task = task_from_turn(turn)
    editing = wants_edit(turn)
    if editing and not (turn.get("request") or turn.get("message")):
        return {"ok": False, "error": "a change request needs a request"}
    if not editing and not task.get("intent"):
        return {"ok": False, "error": "a turn needs a request (or a task with an intent)"}
    owned_node = node is None
    if node is None:
        base_url = turn.get("zetta_base_url") or settings.default_base_url
        token = turn.get("zetta_token") or os.environ.get("ZETTA_TOKEN", "")
        if not base_url or not token:
            return {"ok": False, "error": "no workspace to build against: the turn needs "
                                          "zetta_base_url and zetta_token"}
        node = Node(base_url=base_url, token=token)
    try:
        if editing:
            # agent-project#4: a change is a proposal lake applies on the person's confirmation.
            result = propose(turn, client or OpenAI(), turn.get("model") or settings.model, node)
        else:
            result = build(task, corpus, client or OpenAI(), turn.get("model") or settings.model,
                           node, settings.shots)
    finally:
        if owned_node:
            node.close()
    if not editing:
        result["defects"] = defects(task, result)
        result["applied"] = False
    trace_id = str(uuid.uuid4())
    settings.trace_dir.mkdir(parents=True, exist_ok=True)
    (settings.trace_dir / f"{trace_id}.json").write_text(json.dumps(
        {"trace_id": trace_id, "agent": "agent-project",
         "at": datetime.now(timezone.utc).isoformat(), "turn": {**turn, "zetta_token": None},
         "task": task, "result": result}, indent=1, default=str))
    return {"ok": True, "trace_id": trace_id, "result": result}


def health(settings: Settings, corpus: Corpus) -> tuple[int, dict[str, Any]]:
    """Unhealthy without the corpus: answering zero-shot is a different agent, and a service
    that silently became one would be worse than one that says it cannot start."""
    if not corpus.examples:
        return 503, {"ok": False, "error": f"no corpus at {settings.corpus_path}"}
    return 200, {"ok": True, "agent": "agent-project", "examples": len(corpus.examples),
                 "model": settings.model}


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
            if settings.token and self.headers.get("Authorization") != f"Bearer {settings.token}":
                self._send(401, {"ok": False, "error": "unauthorized"})
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
    ThreadingHTTPServer(("0.0.0.0", port), make_handler(settings, corpus)).serve_forever()
    return 0
