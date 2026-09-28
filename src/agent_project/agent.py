"""The project agent: a profile and an intent in, a Pipeline out — built by calling tools.

No trained model yet. A frontier model, few-shot trajectories retrieved from `agent-project-corpus`,
and the real tool surface. That corpus becomes SFT data later; until then it is the precedent this
agent reasons from, which also keeps it continuously exercised.

What makes this agent different from Auto-Suggest is that it **acts**. Auto-Suggest reads a
situation and proposes; this one calls tools, sees what comes back, and decides the next call from
the observation. The loop enforces that property rather than hoping for it: a call that names no
observation is rejected and the model is told why, because a sequence that would have been
identical without reading the data is a lookup wearing a trajectory's shape.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent_project import catalog, tools, validate
from agent_project.node import Node, describe
from agent_project.retrieval import Corpus

PROMPT_VERSION = "v2"
PROMPTS = Path(__file__).parent / "prompts"
MAX_CALLS = 12


def _shots(examples: list[dict[str, Any]]) -> str:
    """Retrieved examples as precedent — the situation, the trajectory that answered it, and the
    pipeline it produced. The trajectory is the part worth showing: it is what a call justified by
    an observation actually looks like."""
    if not examples:
        return ""
    blocks = []
    for example in examples:
        blocks.append(
            "SITUATION:\n" + json.dumps({"industry": (example.get("profile") or {}).get("industry"),
                                         "intent": example.get("intent"),
                                         "before": example.get("before")}, indent=1)
            + "\nTRAJECTORY:\n" + json.dumps(example.get("trajectory"), indent=1)
            + "\nPIPELINE:\n" + json.dumps(example.get("pipeline"), indent=1))
    return ("\n\nWorked examples from similar situations — precedent, not a template. Their calls "
            "are right for their data; yours must be right for what your calls return:\n\n"
            + "\n\n---\n\n".join(blocks))


def _situation(task: dict[str, Any]) -> dict[str, Any]:
    """What the agent is TOLD, as opposed to what it must find out.

    The task fixture carries the whole profile — every column, every null fraction — because that is
    what an eval needs in order to know the right answer. Handing all of it to the agent makes
    reading unnecessary: it can write a correct-looking trajectory straight from the fixture, and
    the first run did exactly that, opening with `clean` justified by "the profile indicates 5%
    nulls".

    In production no such object exists. The agent is pointed at a node and finds out by asking. So
    the situation it receives names the industry, the persona and WHICH datasets exist — and the
    columns, types and null fractions are withheld. They have to come from `get_dataset`, which is
    what makes the observation load-bearing and the trajectory real.
    """
    profile = task.get("profile") or {}
    return {"industry": profile.get("industry"),
            "persona": profile.get("persona"),
            "datasets": [f"{table.get('database')}.{table.get('name')}"
                         for table in profile.get("tables") or []],
            "intent": task.get("intent"),
            "methodology": task.get("methodology"),
            "before": task.get("before")}


def _finished(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    body = fence.group(1) if fence else text[text.find("{"): text.rfind("}") + 1]
    try:
        answer = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    return answer if isinstance(answer, dict) and "pipeline" in answer else None


def build(task: dict[str, Any], corpus: Corpus, client: OpenAI, model: str, node: Node | None,
          k: int = 2) -> dict[str, Any]:
    """Walk a trajectory and return what was built, alongside the trajectory that built it."""
    # Retrieval reads the FULL profile — it runs locally and never reaches the model, so matching
    # on column names costs the agent nothing it would otherwise have had to discover.
    neighbours = corpus.nearest(task, k)
    system = ((PROMPTS / f"{PROMPT_VERSION}.md").read_text()
              + catalog.as_context() + _shots(neighbours))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(_situation(task))}]

    trajectory: list[dict[str, Any]] = []
    answer: dict[str, Any] | None = None

    for _ in range(MAX_CALLS):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools.specs()).choices[0].message
        messages.append(response.model_dump(exclude_none=True))

        if not response.tool_calls:
            answer = _finished(response.content)
            if answer is not None:
                break
            messages.append({"role": "user", "content":
                             "Return the finished object with `pipeline` and `reply`, or keep "
                             "calling tools."})
            continue

        for call in response.tool_calls:
            try:
                payload = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                payload = {}
            step = {"call": call.function.name, "args": payload.get("args") or {}}
            if payload.get("because"):
                step["because"] = payload["because"]

            # Reject an unjustified call BEFORE running it. Letting it run and noting the defect
            # afterwards would teach the loop that the rule is advisory, and the observation it
            # returned would then justify the next call — laundering the break.
            if trajectory and not step.get("because"):
                messages.append({"role": "tool", "tool_call_id": call.id, "content":
                                 "REJECTED: name the observation that justified this call in "
                                 "`because`. Quote what a previous call returned."})
                continue

            step["observation"] = describe(node, step["call"], step["args"])
            trajectory.append(step)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": str(step["observation"])})

        if answer is not None:
            break

    return {"pipeline": normalize((answer or {}).get("pipeline")),
            "reply": (answer or {}).get("reply"),
            "trajectory": trajectory,
            "few_shot_from": [e.get("example_id", "?") for e in neighbours],
            "prompt_version": PROMPT_VERSION}


TABLE_KEYS = ("training_sources", "datasets", "sources", "training_data")


def normalize(pipeline: Any) -> Any:
    """The fields lake applies a design with, always under the same names (lake applies it to the
    project's pipeline): `dataset` is the one table it trains on, "<database>.<table>", taken from
    the first of `training_sources` (or `datasets`, `sources`) when the model named it there."""
    if not isinstance(pipeline, dict):
        return pipeline
    if not isinstance(pipeline.get("dataset"), str) or not pipeline.get("dataset"):
        for key in TABLE_KEYS:
            tables = pipeline.get(key)
            if isinstance(tables, list) and tables and isinstance(tables[0], str):
                pipeline = {**pipeline, "dataset": tables[0]}
                break
            if isinstance(tables, str) and tables:
                pipeline = {**pipeline, "dataset": tables}
                break
    return pipeline


def defects(task: dict[str, Any], result: dict[str, Any]) -> list[str]:
    """Everything wrong with what was built. Empty means it is usable."""
    problems = tools.undisciplined(result["trajectory"], task.get("before"))
    if result["pipeline"] is None:
        return [*problems, "no pipeline was produced"]

    design = result["pipeline"]
    if isinstance(design, dict) and design.get("model", True):
        # What lake needs to apply the design to a project's pipeline.
        for field in ("dataset", "label", "model_kind"):
            if not design.get(field):
                problems.append(f"the design names no {field}")

    methodology = task.get("methodology")
    if methodology:
        problems += validate.shape_defects(methodology, result["pipeline"])

    # A mutation must change the thing asked for and leave the rest alone.
    before = task.get("before")
    if before:
        changed = {key for key in set(before) | set(result["pipeline"])
                   if before.get(key) != result["pipeline"].get(key)}
        if not changed:
            problems.append("a mutation that changed nothing")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Pipeline from a profile and an intent.")
    parser.add_argument("task", type=Path,
                        help="JSON with a profile, an intent, optionally methodology and before")
    parser.add_argument("--corpus", type=Path,
                        default=Path(os.environ.get("PROJECT_CORPUS", "")))
    parser.add_argument("--model", default=os.environ.get("PROJECT_MODEL", "gpt-4o"))
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--offline", action="store_true",
                        help="describe calls without a live node — for prompt work only")
    args = parser.parse_args()

    task = json.loads(args.task.read_text())
    corpus = Corpus(args.corpus)
    if not corpus.examples:
        print(f"note: no corpus at {args.corpus} — building with no precedent")

    node = None if args.offline else Node()
    try:
        result = build(task, corpus, OpenAI(), args.model, node, args.shots)
    finally:
        if node is not None:
            node.close()

    result["defects"] = defects(task, result)
    print(json.dumps(result, indent=2))
    return 2 if result["defects"] else 0
