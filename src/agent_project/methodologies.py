# SPDX-License-Identifier: MIT

"""The live methodology registry and its one documented compatibility fallback.

The registry belongs to Zetta. This module deliberately contains no portfolio snapshot: every
turn reads the workspace because runnable methods change as the engine gains implementations. The
only local entry is the compatibility contract for engines that answer ``unknown_capability``:
those engines can run M-1 and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from agent_project.node import Node

LEGACY_M1 = {
    "id": "M-1",
    "name": "Churn / GBDT classification",
    "family": "classification",
    "model_kinds": ["gbdt"],
    "inputs": [],
    "outputs": [],
    "metrics": [],
    "runnable": True,
    "partial": False,
}


def _error_code(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    if isinstance(error, dict):
        return error.get("code")
    return value.get("code") if isinstance(value.get("code"), str) else None


def _http_error_code(error: httpx.HTTPStatusError) -> str | None:
    try:
        return _error_code(error.response.json())
    except ValueError:
        return None


@dataclass(frozen=True)
class Registry:
    """The registry observed for one turn."""

    entries: list[dict[str, Any]]
    old_engine: bool = False
    failure: str | None = None

    def entry(self, identifier: str | None) -> dict[str, Any] | None:
        return next((item for item in self.entries if item.get("id") == identifier), None)

    def runnable(self) -> list[dict[str, Any]]:
        return [item for item in self.entries if item.get("runnable") is True]

    def public(self) -> dict[str, Any]:
        value: dict[str, Any] = {"methodologies": self.entries}
        if self.old_engine:
            value["old_engine"] = True
        if self.failure:
            value["failure"] = self.failure
        return value

    def observation(self) -> str:
        if self.old_engine:
            return ("list_methodologies returned unknown_capability. This workspace uses an older "
                    "engine; only M-1 (churn / GBDT classification) is runnable.")
        if self.failure:
            return f"list_methodologies FAILED: {self.failure}"
        return f"list_methodologies returned: {self.public()}"


def read(node: Node | None) -> Registry:
    """Read the turn's registry, recognizing only the specified old-engine response."""
    if node is None:
        return Registry([], failure="OFFLINE: no workspace read ran")
    try:
        value = node.read("list_methodologies")
    except httpx.HTTPStatusError as error:
        if _http_error_code(error) == "unknown_capability":
            return Registry([dict(LEGACY_M1)], old_engine=True)
        return Registry([], failure=f"{error.response.status_code} {error.response.text[:200]}")
    except httpx.HTTPError as error:
        return Registry([], failure=f"{type(error).__name__}: {error}")

    if _error_code(value) == "unknown_capability":
        return Registry([dict(LEGACY_M1)], old_engine=True)
    if not isinstance(value, dict) or not isinstance(value.get("methodologies"), list):
        return Registry([], failure=f"unexpected response {value!r}")
    entries = [item for item in value["methodologies"] if isinstance(item, dict)]
    return Registry(entries)


def decline(registry: Registry, identifier: str | None, reason: str | None = None) -> dict[str, Any]:
    """Build a refusal from registry facts rather than from model-written availability claims."""
    entry = registry.entry(identifier)
    if entry is not None and entry.get("runnable") is not True:
        issue = entry.get("issue")
        why = reason or f"{entry.get('name') or identifier} is not runnable in this workspace"
        return {"methodology": identifier, "issue": issue, "reason": why}
    if registry.old_engine and identifier != "M-1":
        return {
            "methodology": identifier,
            "issue": "workspace engine predates the methodology registry",
            "reason": "this older workspace engine can run only M-1",
        }
    runnable = ", ".join(str(item.get("id")) for item in registry.runnable()) or "none"
    return {
        "methodology": identifier,
        "issue": registry.failure or "methodology is absent from the workspace registry",
        "reason": reason or f"the workspace reports these runnable methodologies: {runnable}",
    }


def decline_reply(value: dict[str, Any]) -> str:
    """A consistent refusal that always carries the engine's issue or compatibility limitation."""
    identifier = value.get("methodology") or "The needed methodology"
    reason = value.get("reason") or "it is not runnable"
    issue = value.get("issue")
    suffix = f" The engine tracks this as {issue}." if issue else ""
    sentence = str(reason).rstrip(".")
    return f"I cannot propose {identifier}: {sentence}.{suffix} Nothing was changed."
