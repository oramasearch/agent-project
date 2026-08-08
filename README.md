# agent-project

Builds and changes Pipelines by **calling tools**. Takes a profile and an intent — the shape
`agent-autosuggest` emits — and walks a trajectory against a live node.

## Not a trained model yet

A frontier model, the solutions catalogue as context, and few-shot trajectories retrieved from
`agent-project-corpus`. That corpus becomes SFT data later; until then it is the precedent this
agent reasons from. See [`zetta-bridge/docs/agent-scaffold.md`](../../docs/agent-scaffold.md).

## Run

```bash
export OPENAI_API_KEY=...
export PROJECT_CORPUS=../agent-project-corpus/corpus/project-sft-v1.jsonl
export ZETTA_BASE_URL=... ZETTA_TOKEN=...
uv run agent-project task.json
```

`task.json` carries a `profile`, an `intent`, optionally a `methodology` and — for a change to an
existing pipeline — a `before`.

`--offline` skips the node for prompt work. Every observation then says so and reports that nothing
ran; it never invents a plausible result, because an offline run must not be mistakable for a real
one.

Exit `0` clean, `2` when what was built has defects (it still prints).

## What makes this a trajectory and not a lookup

Every call after the first names the **observation** that justified it. A call justified by the
user's ask is a scripted step, and a scripted sequence is a lookup with extra steps.

The loop rejects an unjustified call **before running it**. Running it and noting the defect
afterwards would teach the loop that the rule is advisory — and the observation it returned would
then justify the next call, laundering the break.

`tools.undisciplined` also rejects a justification that refers to nothing already observed, which
is what an invented one looks like: *"the user wants churn prediction"* cites the ask, not a result.

## The observations are real

Calls go to the node and the reply is what came back — including when what came back is a failure.
A failing tool is returned as an observation rather than raised, because the agent is supposed to
adapt to what the data actually is, and an exception would end the trajectory instead of informing
it.

The corpus generator learned this the expensive way ([#989]): observations written by the
generating model were coherent by construction, since one author wrote both the observation and the
call responding to it. That is exactly the property a real node does not have.

## Changing an existing pipeline

When a `before` is supplied, **what must not change matters as much as what must**. A mutation that
silently rebuilds has thrown away the user's work even if the result would score well alone — so a
mutation that changed nothing is reported as a defect.

[#989]: https://github.com/oramasearch/zetta/issues/989
