# SPDX-License-Identifier: MIT

"""Edit mode: a change to an existing Pipeline, returned as a proposal lake applies on confirmation.

The Project Agent never saves anything. Its lake key is read-only, and a person must see what will
change before it does (lake#249). So a change request (a turn whose context carries the current
`pipeline` with its `project` and `graph`) answers with a **proposal**: the `edit_pipeline`
operations that would make the change, each checked here against the graph as it will be at that
point, plus the run settings (`dataset`, `holdout`) the request implies, and a reply that says it
is a proposal. lake shows it as a diff and applies it only when the person confirms.

agent-project#4: the agent once answered "updated … in place" to a change it had no way to make.
The contract below makes that answer impossible: the reply is written by this module, from the
validated edits, never by the model.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from openai import OpenAI

from agent_project import methodologies
from agent_project.node import Node, describe

MAX_CALLS = 8

# lake's edit_pipeline operations and the fields each needs (lake src/api/pipeline.cpp).
OPS = {
    "add_step": ("kind", "name"),
    "rename_step": ("step", "name"),
    "kind_step": ("step", "kind"),
    "remove_step": ("step",),
    "add_edge": ("from", "to"),
    "remove_edge": ("from", "to"),
    "set_attribute": ("step", "name"),
    "rename": ("name",),
}
STEP_KINDS = ("connect", "clean", "model", "index", "serve", "act")
EDGE_KINDS = ("data_flow", "control")

PROMPT = """You change an existing Zetta Pipeline (an execution graph of steps joined by edges) at a
person's request. You do NOT save anything: you return the edits for the person to review and apply.

The runtime has already read `list_methodologies`. The user message includes that live registry.
Choose from it, never from memory. If the requested methodology is not runnable, return a decline
that names its id, issue and what must become runnable. Never reshape another methodology into an
unsupported substitute.

Before proposing graph edits, call `analyze_dataset`, then `plan_features` with the chosen
methodology, then `compile_intent` with project and methodology. Treat the compiled graph as
the required shape. Your edits must make the current graph match it, including the model step's
model_kind and any methodology-specific steps. If feature planning says transform first, include
that transform in the proposal instead of featurizing the raw data badly.

The steps' kinds, in order: connect, clean, model, index, serve, act. An edge carries data_flow or
only orders steps (control). Every step has an id, a kind, a name and attributes (for example a
connect step's `source` and `table`, a model step's `target` and `algo`).

The operations you may propose, applied in order:
- {"op":"add_step","kind":K,"name":N,"id":ID}  (ID is the id you will use for it in later edits)
- {"op":"rename_step","step":ID,"name":N}
- {"op":"kind_step","step":ID,"kind":K}
- {"op":"remove_step","step":ID}  (its edges go with it)
- {"op":"add_edge","from":ID,"to":ID,"kind":"data_flow"|"control"}
- {"op":"remove_edge","from":ID,"to":ID}
- {"op":"set_attribute","step":ID,"name":KEY,"value":VALUE}  (a null value removes it)
- {"op":"rename","name":N}  (the pipeline)

Read before you change: call list_datasets and get_dataset to learn what data really exists, and
point connect steps at a dataset that exists (set_attribute source "datasets", table <name>). Never
invent a source. Run settings are not graph edits: put the dataset a run should train on
("<database>.<table>") and the hold-out fraction (0..0.5) in "run".

For a runnable change, answer with JSON only:
{"methodology":"M-n", "model_kind":"a registry kind", "summary":"what changes", "edits":[...],
 "run":{"dataset":..., "holdout":...}}

For an unsupported change, answer with JSON only:
{"declined":{"methodology":"M-n", "issue":"the registry issue", "reason":"what must run"}}
"""

WORKSPACE_TOOLS = [
    {"type": "function", "function": {
        "name": "list_datasets", "description": "The datasets in the workspace. No arguments.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_dataset", "description": "One dataset's columns and rows. Args: dataset.",
        "parameters": {"type": "object", "properties": {"dataset": {"type": "string"}},
                       "required": ["dataset"]}}},
    {"type": "function", "function": {
        "name": "analyze_dataset",
        "description": "Semantic types, grain, keys, joins and leakage suspects.",
        "parameters": {"type": "object", "properties": {"dataset": {"type": "string"}},
                       "required": ["dataset"]}}},
    {"type": "function", "function": {
        "name": "plan_features", "description": "Plan features or report transform first.",
        "parameters": {"type": "object", "properties": {
            "dataset": {"type": "string"}, "methodology": {"type": "string"}},
            "required": ["dataset", "methodology"]}}},
    {"type": "function", "function": {
        "name": "compile_intent", "description": "Compile the methodology's engine-owned graph.",
        "parameters": {"type": "object", "properties": {
            "project": {"type": "string"}, "methodology": {"type": "string"},
            "hints": {"type": "object"}},
            "required": ["project", "methodology"]}}},
]


def wants_edit(turn: dict[str, Any]) -> bool:
    context = turn.get("context") or {}
    return isinstance(context.get("pipeline"), dict) and bool(context.get("project"))


def simulate(pipeline: dict[str, Any], edits: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Apply the edits to a copy of the graph the way lake will, and say what is wrong with them.

    Returns the graph after every edit that applied, and one problem per edit that would not.
    """
    graph = copy.deepcopy(pipeline)
    nodes: list[dict[str, Any]] = graph.setdefault("nodes", [])
    edges: list[dict[str, Any]] = graph.setdefault("edges", [])
    problems: list[str] = []

    def step(identifier: str) -> dict[str, Any] | None:
        return next((node for node in nodes if node.get("id") == identifier), None)

    for index, edit in enumerate(edits, start=1):
        op = edit.get("op")
        missing = [field for field in OPS.get(op, ()) if not edit.get(field)]
        if op not in OPS:
            problems.append(f"edit {index}: unknown op {op!r}")
            continue
        if missing:
            problems.append(f"edit {index} ({op}): missing {', '.join(missing)}")
            continue
        if op == "add_step":
            if edit["kind"] not in STEP_KINDS:
                problems.append(f"edit {index}: {edit['kind']!r} is not a step kind")
                continue
            identifier = edit.get("id") or f"{edit['kind']}_{len(nodes) + 1}"
            if step(identifier):
                problems.append(f"edit {index}: a step {identifier!r} already exists")
                continue
            edit["id"] = identifier
            nodes.append({"id": identifier, "kind": edit["kind"], "name": edit["name"],
                          "attributes": {}})
        elif op in ("rename_step", "kind_step", "remove_step", "set_attribute"):
            target = step(edit["step"])
            if target is None:
                problems.append(f"edit {index} ({op}): there is no step {edit['step']!r}")
                continue
            if op == "rename_step":
                target["name"] = edit["name"]
            elif op == "kind_step":
                if edit["kind"] not in STEP_KINDS:
                    problems.append(f"edit {index}: {edit['kind']!r} is not a step kind")
                    continue
                target["kind"] = edit["kind"]
            elif op == "remove_step":
                nodes.remove(target)
                edges[:] = [e for e in edges if edit["step"] not in (e.get("from"), e.get("to"))]
            else:
                attributes = target.setdefault("attributes", {})
                if edit.get("value") is None:
                    attributes.pop(edit["name"], None)
                else:
                    attributes[edit["name"]] = edit["value"]
        elif op == "add_edge":
            kind = edit.get("kind") or "data_flow"
            if kind not in EDGE_KINDS:
                problems.append(f"edit {index}: {kind!r} is not an edge kind")
                continue
            if not step(edit["from"]) or not step(edit["to"]) or edit["from"] == edit["to"]:
                problems.append(f"edit {index}: an edge must join two existing, different steps")
                continue
            if any(e.get("from") == edit["from"] and e.get("to") == edit["to"] for e in edges):
                problems.append(f"edit {index}: that edge already exists")
                continue
            edges.append({"from": edit["from"], "to": edit["to"], "kind": kind})
        elif op == "remove_edge":
            before = len(edges)
            edges[:] = [e for e in edges
                        if not (e.get("from") == edit["from"] and e.get("to") == edit["to"])]
            if len(edges) == before:
                problems.append(f"edit {index}: there is no edge {edit['from']} → {edit['to']}")
        elif op == "rename":
            graph["name"] = edit["name"]
    return graph, problems


def plain(edit: dict[str, Any]) -> str:
    """One edit in words a person reads before applying it."""
    op = edit["op"]
    if op == "add_step":
        return f"Add a {edit['kind']} step “{edit['name']}”"
    if op == "rename_step":
        return f"Rename step {edit['step']} to “{edit['name']}”"
    if op == "kind_step":
        return f"Make step {edit['step']} a {edit['kind']} step"
    if op == "remove_step":
        return f"Remove step {edit['step']} and its edges"
    if op == "add_edge":
        return f"Connect {edit['from']} → {edit['to']} ({edit.get('kind') or 'data_flow'})"
    if op == "remove_edge":
        return f"Disconnect {edit['from']} → {edit['to']}"
    if op == "set_attribute":
        if edit.get("value") is None:
            return f"Clear {edit['name']} on step {edit['step']}"
        return f"Set {edit['name']} = {json.dumps(edit['value'])} on step {edit['step']}"
    return f"Rename the pipeline to “{edit['name']}”"


def _answer(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    body = fence.group(1) if fence else text[text.find("{"): text.rfind("}") + 1]
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("declined"), dict):
        return value
    return value if isinstance(value.get("edits"), list) else None


def _declined(registry: methodologies.Registry, requested: dict[str, Any],
              reads: list[dict[str, Any]]) -> dict[str, Any]:
    value = methodologies.decline(
        registry,
        requested.get("methodology"),
        requested.get("reason"),
    )
    return {
        "kind": "declined",
        "proposal": None,
        "declined": value,
        "reads": reads,
        "reply": methodologies.decline_reply(value),
    }


def propose(turn: dict[str, Any], client: OpenAI, model: str, node: Node | None) -> dict[str, Any]:
    """Read what it needs, then return a checked proposal. Nothing is saved."""
    context = turn.get("context") or {}
    pipeline = context["pipeline"]
    request = turn.get("request") or turn.get("message") or ""
    registry = methodologies.read(node)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps({
            "request": request, "project": context.get("project"),
            "methodology_registry": registry.public(),
            "registry_observation": registry.observation(),
            "pipeline": {"name": pipeline.get("name"), "nodes": pipeline.get("nodes"),
                         "edges": pipeline.get("edges")}})}]
    reads: list[dict[str, Any]] = [{
        "call": "list_methodologies",
        "args": {},
        "observation": registry.observation(),
    }]
    answer: dict[str, Any] | None = None
    for _ in range(MAX_CALLS):
        message = client.chat.completions.create(
            model=model, messages=messages, tools=WORKSPACE_TOOLS).choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            answer = _answer(message.content)
            if answer is not None:
                break
            messages.append({"role": "user", "content": "Answer with the JSON object only."})
            continue
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            observation = describe(node, call.function.name, args)
            reads.append({"call": call.function.name, "args": args,
                          "observation": observation})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})

    if answer is None:
        return {"kind": "proposal", "proposal": None, "reads": reads,
                "reply": "I could not work out a change to propose. Nothing was changed."}

    if isinstance(answer.get("declined"), dict):
        return _declined(registry, answer["declined"], reads)

    identifier = answer.get("methodology")
    entry = registry.entry(identifier)
    if entry is None or entry.get("runnable") is not True:
        return _declined(registry, {"methodology": identifier}, reads)

    edits = [edit for edit in answer.get("edits") or [] if isinstance(edit, dict)]
    after, problems = simulate(pipeline, edits)
    run = answer.get("run") if isinstance(answer.get("run"), dict) else {}
    holdout = run.get("holdout")
    if holdout is not None and not (isinstance(holdout, (int, float)) and 0 <= holdout <= 0.5):
        problems.append(f"holdout {holdout!r} must be a fraction between 0 and 0.5")
    model_kinds = entry.get("model_kinds") or []
    if not answer.get("model_kind"):
        problems.append("the proposal names no model_kind")
    elif model_kinds and answer["model_kind"] not in model_kinds:
        problems.append(f"{answer['model_kind']!r} is not a model kind for {identifier}")
    calls = {item.get("call") for item in reads}
    for required in ("analyze_dataset", "plan_features", "compile_intent"):
        if required not in calls:
            problems.append(f"the proposal did not call {required}")
    for item in reads:
        if item.get("call") in {"plan_features", "compile_intent"}:
            used = item.get("args", {}).get("methodology")
            if used != identifier:
                problems.append(f"{item['call']} used {used!r}, not {identifier}")
        if item.get("call") in {"analyze_dataset", "plan_features", "compile_intent"}:
            observation = str(item.get("observation") or "")
            if "FAILED:" in observation or "not_supported" in observation:
                problems.append(f"{item['call']} did not succeed: {observation}")
    feature_plan = next((item.get("observation") for item in reversed(reads)
                         if item.get("call") == "plan_features"), None)
    proposal = {"project": context.get("project"), "graph": context.get("graph") or pipeline.get("name"),
                "methodology": identifier, "model_kind": answer.get("model_kind"),
                "feature_plan": feature_plan,
                "summary": answer.get("summary") or "", "edits": edits,
                "steps": [plain(edit) for edit in edits], "run": run, "problems": problems,
                "after": {"nodes": after.get("nodes"), "edges": after.get("edges"),
                          "name": after.get("name")}}
    if problems:
        reply = ("I proposed a change, but it would not apply as written: " + "; ".join(problems)
                 + ". Nothing was changed.")
    elif not edits and not run:
        reply = "The pipeline already does what you asked; I propose no change. Nothing was changed."
    else:
        reply = ("Proposed, not applied: " + (answer.get("summary") or "a change") + ". Review "
                 "the " + str(len(edits)) + " edit(s) and apply them on the pipeline page.")
    return {"kind": "proposal", "proposal": proposal, "reads": reads, "reply": reply}
