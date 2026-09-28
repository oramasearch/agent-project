"""The live solution registry and the tools used to operate one solution.

The catalogue belongs to the workspace.  This module keeps no S- or B-number snapshot: every
turn reads ``list_solutions`` and works from the entries that response supplied.  An engine that
answers ``unknown_capability`` predates the registry and must use the methodology path instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from agent_project.node import Node

PROMPT_VERSION = "v4"
PROMPT = Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md"
MAX_CALLS = 12

READS = {
    "get_solution": "Read one live solution entry. Args: solution.",
    "get_policy": "Read the current decision policy. Args come from the workspace contract.",
    "get_experiment": "Read holdout assignment and balance. Args come from the workspace contract.",
    "list_ledger": "Read the solution's action and outcome evidence. Args are optional filters.",
    "get_monitor": "Read drift and guardrail state. Args come from the workspace contract.",
    "get_roi_readout": "Read the measured solution readout. Args come from the workspace contract.",
}

ACTIONS = {
    "create_solution": (
        "Build a catalogue solution. Args: project, solution, level (1-4), "
        "sources {source, table}."
    ),
    "run_solution": "Run a built solution graph. Args: project, graph.",
    "decide": "Apply an approved policy to scores. Args come from the workspace contract.",
    "approve_decision": (
        "Approve one pending A1 decision after explicit owner approval. "
        "Args come from the workspace contract."
    ),
}


def _error_code(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    if isinstance(error, dict):
        return error.get("code")
    code = value.get("code")
    return code if isinstance(code, str) else None


def _http_error_code(error: httpx.HTTPStatusError) -> str | None:
    try:
        return _error_code(error.response.json())
    except ValueError:
        return None


@dataclass(frozen=True)
class Registry:
    """The solution entries observed at the beginning of one turn."""

    entries: list[dict[str, Any]]
    old_engine: bool = False
    failure: str | None = None

    def entry(self, identifier: str | None) -> dict[str, Any] | None:
        return next((item for item in self.entries if item.get("id") == identifier), None)

    def public(self) -> dict[str, Any]:
        value: dict[str, Any] = {"solutions": self.entries}
        if self.old_engine:
            value["old_engine"] = True
        if self.failure:
            value["failure"] = self.failure
        return value

    def observation(self) -> str:
        if self.old_engine:
            return (
                "list_solutions returned unknown_capability. This workspace uses an older "
                "engine, so solution operations are unavailable; use methodologies only."
            )
        if self.failure:
            return f"list_solutions FAILED: {self.failure}"
        return f"list_solutions returned: {self.public()}"


def read(node: Node | None) -> Registry:
    """Read this turn's solution registry without inventing an empty catalogue."""
    if node is None:
        return Registry([], failure="OFFLINE: no workspace read ran")
    try:
        value = node.read("list_solutions")
    except httpx.HTTPStatusError as error:
        if _http_error_code(error) == "unknown_capability":
            return Registry([], old_engine=True)
        return Registry([], failure=f"{error.response.status_code} {error.response.text[:200]}")
    except httpx.HTTPError as error:
        return Registry([], failure=f"{type(error).__name__}: {error}")

    if _error_code(value) == "unknown_capability":
        return Registry([], old_engine=True)
    if not isinstance(value, dict) or not isinstance(value.get("solutions"), list):
        return Registry([], failure=f"unexpected response {value!r}")
    entries = [item for item in value["solutions"] if isinstance(item, dict)]
    return Registry(entries)


def specs() -> list[dict[str, Any]]:
    """Solution tools in the OpenAI SDK shape; policy mutation is intentionally absent."""
    result = []
    for name, description in {**READS, **ACTIONS}.items():
        result.append(
            {
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
                                "description": (
                                    "Quote the previous observation that justifies this call."
                                ),
                            },
                        },
                        "required": ["args", "because"],
                    },
                },
            }
        )
    return result


def _answer(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    body = fence.group(1) if fence else text[text.find("{"): text.rfind("}") + 1]
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _levels(entry: dict[str, Any]) -> set[int]:
    raw = entry.get("levels") or entry.get("capability_levels") or []
    result = set()
    for item in raw:
        value = item.get("level") if isinstance(item, dict) else item
        if isinstance(value, int) and not isinstance(value, bool):
            result.add(value)
    return result


def _missing(entry: dict[str, Any], created: dict[str, Any] | None) -> list[dict[str, Any]]:
    combined = entry.get("missing") or entry.get("missing_components") or []
    if isinstance(created, dict):
        combined = created.get("missing") or created.get("missing_components") or combined
    return [item for item in combined if isinstance(item, dict)]


def _policy_proposal(entry: dict[str, Any]) -> dict[str, Any]:
    template = entry.get("policy") or entry.get("decision_policy_template")
    autonomy = entry.get("autonomy") or entry.get("autonomy_defaults") or []
    return {
        "status": "proposed",
        "approval_required": True,
        "definition": template,
        "autonomy": autonomy,
        "guardrails": entry.get("guardrails") or [],
    }


def _label_definitions(entry: dict[str, Any]) -> list[dict[str, Any]]:
    roles = entry.get("data_roles") or entry.get("required_data_roles") or []
    definitions = []
    for role in roles:
        if not isinstance(role, dict):
            continue
        definitions.append({
            "role": role.get("role"),
            "required": role.get("required") is True,
            "definition": role.get("description") or "",
        })
    measurement = entry.get("measurement") or entry.get("measurement_design")
    if measurement:
        definitions.append({
            "role": "measured_outcome",
            "required": True,
            "definition": measurement,
        })
    return definitions


def _approval_matches(task: dict[str, Any], args: dict[str, Any]) -> bool:
    approval = task.get("approval")
    if not isinstance(approval, dict) or approval.get("approved") is not True:
        return False
    expected = approval.get("decision_id") or approval.get("decision")
    supplied = args.get("decision_id") or args.get("decision")
    return bool(expected and supplied and expected == supplied)


def _observed_entry(identifier: str | None,
                    trajectory: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in reversed(trajectory):
        if step.get("call") != "get_solution":
            continue
        if step.get("args", {}).get("solution") != identifier:
            continue
        value = step.get("value")
        if not isinstance(value, dict):
            continue
        detailed = value.get("solution")
        entry = detailed if isinstance(detailed, dict) else value
        if entry.get("id") == identifier:
            return entry
    return None


def _justified(because: str, trajectory: list[dict[str, Any]]) -> bool:
    prior_calls = {str(step.get("call")) for step in trajectory}
    if any(name in because for name in prior_calls):
        return True
    words = {word.strip(".,;:()[]'\"`").lower() for word in because.split()}
    observations = " ".join(str(step.get("observation") or "") for step in trajectory)
    seen = {word.strip(".,;:()[]'\"`").lower() for word in observations.split()}
    stop = {"the", "a", "an", "is", "are", "was", "that", "this", "it", "and", "to",
            "of", "in", "for", "on", "with", "returned", "showed"}
    return bool((words - stop) & (seen - stop))


def _preflight(name: str, args: dict[str, Any], task: dict[str, Any], registry: Registry,
               trajectory: list[dict[str, Any]]) -> str | None:
    if name not in READS and name not in ACTIONS:
        return f"{name!r} is not a solution capability"
    identifier = args.get("solution")
    if name in {"get_solution", "create_solution"}:
        entry = registry.entry(identifier)
        if entry is None:
            return f"{identifier!r} is absent from this turn's solution registry"
    if name == "create_solution":
        entry = _observed_entry(identifier, trajectory) or registry.entry(identifier) or {}
        if entry.get("status") == "sketch":
            return f"{identifier} is a sketch and cannot be built"
        if _observed_entry(identifier, trajectory) is None:
            return f"get_solution must read {identifier} before create_solution"
        level = args.get("level")
        if not isinstance(level, int) or isinstance(level, bool):
            return "create_solution level must be an integer"
        levels = _levels(entry)
        if levels and level not in levels:
            return f"level {level} is absent from {identifier}'s live levels"
        sources = args.get("sources")
        if not isinstance(sources, dict) or not sources.get("source") or not sources.get("table"):
            return "create_solution sources must name source and table"
    if name == "run_solution":
        created = next(
            (step for step in reversed(trajectory)
             if step.get("call") == "create_solution" and isinstance(step.get("value"), dict)),
            None,
        )
        if created is None:
            return "run_solution needs a successful create_solution observation"
        value = created["value"]
        if args.get("project") != value.get("project") or args.get("graph") != value.get("graph"):
            return "run_solution project and graph must match create_solution"
    if name == "approve_decision" and not _approval_matches(task, args):
        return "approve_decision needs matching explicit owner approval in the turn context"
    return None


def _invoke(node: Node | None, name: str, args: dict[str, Any]) -> tuple[Any, str]:
    if node is None:
        return None, f"OFFLINE: {name} did not run"
    try:
        if name in READS:
            value = node.read(name, **args)
        else:
            value = node.act(name, **args)
    except httpx.HTTPStatusError as error:
        code = _http_error_code(error)
        if name in READS and name != "get_solution" and code == "unknown_capability":
            return None, f"{name} is not deployed yet (unknown_capability)"
        return None, f"{name} FAILED: {error.response.status_code} {error.response.text[:200]}"
    except httpx.HTTPError as error:
        return None, f"{name} FAILED: {type(error).__name__}: {error}"
    code = _error_code(value)
    if code == "unknown_capability" and name in READS and name != "get_solution":
        return None, f"{name} is not deployed yet (unknown_capability)"
    if code:
        return None, f"{name} FAILED: {code} {value}"
    return value, f"{name} returned: {value}"


def _decline(registry: Registry, identifier: str | None, reason: str | None = None) -> dict[str, Any]:
    entry = registry.entry(identifier)
    issue = entry.get("issue") if entry else None
    if entry and entry.get("status") == "sketch":
        reason = entry.get("promotion_condition") or reason or "the module is still a sketch"
    else:
        reason = reason or "the workspace did not build this solution"
    suffix = f" The engine tracks this as {issue}." if issue else ""
    reply = f"I cannot build {identifier or 'that solution'}: {str(reason).rstrip('.')}.{suffix}"
    return {
        "kind": "declined",
        "solution": None,
        "declined": {"solution": identifier, "issue": issue, "reason": reason},
        "reply": reply,
    }


def _operation_values(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for step in trajectory:
        name = step.get("call")
        if name in READS or name in {"run_solution", "decide", "approve_decision"}:
            result[name] = {
                "value": step.get("value"),
                "observation": step.get("observation"),
            }
    return result


def _readout(operations: dict[str, Any], explanation: str) -> dict[str, Any]:
    observed = operations.get("get_roi_readout") or {}
    value = observed.get("value")
    observation = str(observed.get("observation") or "")
    if value is not None:
        status = "available"
    elif "not deployed yet" in observation:
        status = "not_deployed_yet"
    else:
        status = "unavailable"
    return {
        "status": status,
        "value": value,
        "explanation": explanation,
    }


def _solution_result(task: dict[str, Any], registry: Registry, trajectory: list[dict[str, Any]],
                     answer: dict[str, Any]) -> dict[str, Any]:
    created_step = next(
        (step for step in reversed(trajectory)
         if step.get("call") == "create_solution" and isinstance(step.get("value"), dict)),
        None,
    )
    if created_step is None:
        policy_change = answer.get("policy_change")
        if isinstance(policy_change, dict):
            identifier = policy_change.get("solution") or task.get("solution")
            entry = _observed_entry(identifier, trajectory) or registry.entry(identifier) or {}
            operations = _operation_values(trajectory)
            current = (operations.get("get_policy") or {}).get("value")
            policy_change = {**policy_change, "before": current}
            return {
                "kind": "policy_proposal",
                "solution": {"id": identifier},
                "policy_change": policy_change,
                "policy_proposal": _policy_proposal(entry),
                "approval_required": True,
                "operations": operations,
                "reply": (
                    "Proposed, not applied: review this policy change and approve it as the "
                    "solution owner."
                ),
            }
        requested = answer.get("declined") or {}
        identifier = requested.get("solution") or task.get("solution")
        return _decline(registry, identifier, requested.get("reason"))
    args = created_step.get("args") or {}
    created = created_step.get("value") if isinstance(created_step.get("value"), dict) else {}
    identifier = args.get("solution")
    entry = _observed_entry(identifier, trajectory) or registry.entry(identifier) or {}
    missing = _missing(entry, created)
    graph = created.get("graph")
    preview = created.get("preview") is True
    ready = entry.get("runnable") is True and not missing
    reply = str(answer.get("reply") or "").strip()
    operations = _operation_values(trajectory)
    readout = _readout(operations, reply)
    if missing:
        issues = ", ".join(str(item.get("issue") or item.get("name")) for item in missing)
        reply = (reply + " " if reply else "") + (
            f"The pipeline builds as a preview; production operation needs {issues}."
        )
    reply = (reply + " " if reply else "") + (
        "The policy and label definitions below require owner sign-off before production use."
    )
    return {
        "kind": "solution",
        "solution": {
            "id": identifier,
            "project": args.get("project"),
            "level": args.get("level"),
            "sources": args.get("sources"),
            "graph": graph,
            "preview": preview,
            "production_ready": ready,
            "missing": missing,
        },
        "policy_proposal": _policy_proposal(entry),
        "label_definitions": _label_definitions(entry),
        "owner_signoff": {
            "required": True,
            "items": ["policy_proposal", "label_definitions"],
        },
        "operations": operations,
        "readout": readout,
        "reply": reply,
    }


def unavailable(registry: Registry, identifier: str | None = None) -> dict[str, Any]:
    """A solution request that cannot reach a solution registry."""
    if registry.old_engine:
        reason = "this older workspace engine has no solution registry; methodologies only"
    else:
        reason = registry.failure or "the workspace returned no solution entries"
    result = _decline(registry, identifier, reason)
    result["solution_registry"] = registry.public()
    result["trajectory"] = [{
        "call": "list_solutions",
        "args": {},
        "observation": registry.observation(),
    }]
    result["prompt_version"] = PROMPT_VERSION
    return result


def operate(task: dict[str, Any], client: OpenAI, model: str, node: Node | None,
            registry: Registry) -> dict[str, Any]:
    """Build, run and read one solution while retaining every workspace observation."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": PROMPT.read_text()},
        {"role": "user", "content": json.dumps({
            "mode": "solution",
            "request": task.get("intent"),
            "project": task.get("project"),
            "requested_solution": task.get("solution"),
            "requested_level": task.get("level"),
            "requested_sources": task.get("sources"),
            "solution_registry": registry.public(),
            "registry_observation": registry.observation(),
        })},
    ]
    trajectory: list[dict[str, Any]] = [{
        "call": "list_solutions",
        "args": {},
        "observation": registry.observation(),
    }]
    answer = None
    for _ in range(MAX_CALLS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=specs(),
        ).choices[0].message
        messages.append(response.model_dump(exclude_none=True))
        if not response.tool_calls:
            answer = _answer(response.content)
            if answer is not None:
                break
            messages.append({"role": "user", "content": "Return the finished JSON object."})
            continue
        for call in response.tool_calls:
            try:
                payload = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                payload = {}
            args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
            step = {"call": call.function.name, "args": args}
            because = payload.get("because")
            if because:
                step["because"] = because
            if not because:
                observation = "REJECTED: quote a previous observation in because"
            elif not _justified(because, trajectory):
                observation = "REJECTED: because does not refer to a previous observation"
            else:
                problem = _preflight(call.function.name, args, task, registry, trajectory)
                if problem:
                    observation = f"REJECTED: {problem}"
                else:
                    value, observation = _invoke(node, call.function.name, args)
                    step["value"] = value
            step["observation"] = observation
            trajectory.append(step)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": observation,
            })
    result = _solution_result(task, registry, trajectory, answer or {})
    result["solution_registry"] = registry.public()
    result["trajectory"] = trajectory
    result["prompt_version"] = PROMPT_VERSION
    return result


def defects(result: dict[str, Any]) -> list[str]:
    """Check that a solution answer is supported by its registry and operations."""
    problems = []
    trajectory = result.get("trajectory") or []
    if not trajectory or trajectory[0].get("call") != "list_solutions":
        problems.append("the solution turn did not start with list_solutions")
    for step in trajectory[1:]:
        if not step.get("because"):
            problems.append(f"{step.get('call')} names no prior observation")
        observation = str(step.get("observation") or "")
        expected_refusal = result.get("kind") == "declined" and "not_supported" in observation
        if "REJECTED:" in observation or ("FAILED:" in observation and not expected_refusal):
            problems.append(f"{step.get('call')} did not succeed: {observation}")
    if result.get("kind") == "declined":
        declined = result.get("declined") or {}
        registry = Registry((result.get("solution_registry") or {}).get("solutions") or [])
        entry = registry.entry(declined.get("solution"))
        if entry and entry.get("status") == "sketch":
            condition = str(entry.get("promotion_condition") or "")
            if condition and condition not in str(result.get("reply") or ""):
                problems.append("the sketch refusal does not name its promotion condition")
        return problems
    if result.get("kind") == "policy_proposal":
        calls = {step.get("call") for step in trajectory}
        for name in ("get_solution", "get_policy"):
            if name not in calls:
                problems.append(f"the policy proposal did not call {name}")
        if result.get("approval_required") is not True:
            problems.append("the policy change does not require owner approval")
        if "update_policy" in calls or "create_policy" in calls:
            problems.append("the agent attempted to mutate policy before approval")
        return problems
    solution = result.get("solution") or {}
    registry = Registry((result.get("solution_registry") or {}).get("solutions") or [])
    entry = registry.entry(solution.get("id"))
    if entry is None:
        problems.append("the built solution is absent from this turn's registry")
    calls = {step.get("call") for step in trajectory}
    required = {"get_solution", "create_solution", "run_solution", "get_policy",
                "get_experiment", "list_ledger", "get_monitor", "get_roi_readout"}
    for name in sorted(required - calls):
        problems.append(f"the solution turn did not call {name}")
    proposal = result.get("policy_proposal") or {}
    if proposal.get("approval_required") is not True:
        problems.append("the policy proposal does not require owner approval")
    if not isinstance(result.get("label_definitions"), list):
        problems.append("the solution turn returned no label definitions")
    signoff = result.get("owner_signoff") or {}
    if signoff.get("required") is not True:
        problems.append("the policy and label definitions do not require owner sign-off")
    readout = result.get("readout") or {}
    if not readout.get("explanation"):
        problems.append("the measured readout has no explanation")
    for item in solution.get("missing") or []:
        issue = str(item.get("issue") or "")
        if issue and issue not in str(result.get("reply") or ""):
            problems.append(f"the production caveat does not name {issue}")
    return problems
