"""Read Lake App state and prepare ApplicationAgent handoffs without writing anything.

Apps and builds belong to Lake rather than the workspace Zetta endpoint.  A Project Agent turn
receives a short-lived ``lake_token`` and ``lake_url`` alongside its workspace credential.  This
module uses that Lake credential only for the six read capabilities the Apps contract names.
App, build, and agent-turn actions remain proposals for a person to confirm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


READS = {
    "list_apps": "List the customer's Apps and their desired and observed state.",
    "get_app": "Read one App by app_id.",
    "receive_app_events": "Read retained App events by app_id and sequence.",
    "list_builds": "List the customer's image builds.",
    "get_build": "Read one image build by build_id.",
    "receive_build_events": "Read retained build events by build_id and sequence.",
}

PROPOSED_ONLY_ACTIONS = frozenset({
    "agent_turn",
    "create_app",
    "deploy_app",
    "rollback_app",
    "delete_app",
    "set_app_env",
    "unset_app_env",
    "set_app_auth",
    "restart_app",
    "stop_app",
    "start_app",
    "scale_app",
    "set_app_zetta_binding",
    "bind_service_to_app",
    "unbind_service_from_app",
    "create_service_and_bind_to_app",
    "build_app",
    "build_image",
    "cancel_build",
})


class LakeReader(Protocol):
    """The read half of Lake used by the Project Agent."""

    def read(self, capability: str, **params: Any) -> Any:
        """Call one Lake read capability."""


class Lake:
    """A short-lived Lake client backed by the task's delegated credential."""

    def __init__(self, base_url: str, token: str) -> None:
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=180.0,
        )

    def read(self, capability: str, **params: Any) -> Any:
        response = self.client.get(f"/v1/{capability}", params=params)
        response.raise_for_status()
        body = response.json()
        return body.get("value", body)

    def close(self) -> None:
        self.client.close()


@dataclass(frozen=True)
class Evidence:
    """The exact App and build observations collected for one handoff."""

    reads: list[dict[str, Any]]

    def public(self) -> dict[str, Any]:
        return {"reads": self.reads}


def specs() -> list[dict[str, Any]]:
    """The executable App tool surface: reads only, with no mutation capability."""
    result = []
    for name, description in READS.items():
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "object",
                            "description": "Arguments named in the capability description.",
                        },
                        "because": {
                            "type": "string",
                            "description": "Quote the prior observation that justifies this read.",
                        },
                    },
                    "required": ["args", "because"],
                },
            },
        })
    return result


def _error(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    if isinstance(error, dict):
        return error
    if value.get("ok") is False:
        return value
    return None


def _read(lake: LakeReader, capability: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        value = lake.read(capability, **params)
    except httpx.HTTPStatusError as error:
        return {
            "capability": capability,
            "args": params,
            "ok": False,
            "error": f"{error.response.status_code} {error.response.text[:200]}",
        }
    except httpx.HTTPError as error:
        return {
            "capability": capability,
            "args": params,
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }

    error = _error(value)
    if error is not None:
        return {
            "capability": capability,
            "args": params,
            "ok": False,
            "error": error,
        }
    return {
        "capability": capability,
        "args": params,
        "ok": True,
        "value": value,
    }


def read_state(lake: LakeReader, context: dict[str, Any]) -> Evidence:
    """Read list state and any selected App/build plus their retained events."""
    customer_id = context.get("customer_id")
    if not isinstance(customer_id, str) or not customer_id:
        raise ValueError("an App handoff needs customer_id")

    common = {"customer_id": customer_id}
    reads = [_read(lake, "list_apps", common)]

    app_id = context.get("app_id")
    app = context.get("app")
    if not app_id and isinstance(app, dict):
        app_id = app.get("id") or app.get("app_id")
    if isinstance(app_id, str) and app_id:
        app_args = {**common, "app_id": app_id}
        reads.append(_read(lake, "get_app", app_args))
        event_args = {
            **app_args,
            "from_sequence": context.get("app_from_sequence", 1),
            "wait_ms": 0,
        }
        reads.append(_read(lake, "receive_app_events", event_args))

    reads.append(_read(lake, "list_builds", common))

    build_id = context.get("build_id")
    build = context.get("build")
    if not build_id and isinstance(build, dict):
        build_id = build.get("id") or build.get("build_id")
    if isinstance(build_id, str) and build_id:
        build_args = {**common, "build_id": build_id}
        reads.append(_read(lake, "get_build", build_args))
        event_args = {
            **build_args,
            "from_sequence": context.get("build_from_sequence", 1),
            "wait_ms": 0,
        }
        reads.append(_read(lake, "receive_build_events", event_args))

    return Evidence(reads)


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def application_agent_proposal(
        project: str,
        tables: list[Any],
        model: Any,
        results: dict[str, Any],
        evidence: Evidence) -> dict[str, Any]:
    """Return the exact agent_turn a person may confirm; do not start it here."""
    context = {
        "project": project,
        "tables": tables,
        "model": model,
        "results": results,
    }
    message = (
        f"Build a data app for project {project} over result tables {_json_value(tables)}. "
        f"Use the project's model metadata {_json_value(model)} and preserve the result meaning "
        "described in context. Return an ApplicationAgent proposal for review; do not deploy it."
    )
    return {
        "kind": "proposal",
        "capability": "agent_turn",
        "agent": "agent-application",
        "turn_kind": "app",
        "confirmation_required": True,
        "request": {
            "agent": "agent-application",
            "kind": "app",
            "message": message,
            "context": context,
        },
        "inputs": context,
        "evidence": evidence.public(),
    }
