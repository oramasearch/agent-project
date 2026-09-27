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

## Service mode (in lake)

In lake the Project Agent is a specialized agent of the **Agent Library**: Design's *Create a
Solution* and Ask Orama (as a subagent) send it turns
([lake#228](https://github.com/oramasearch/lake/issues/228)). `agent-project-serve` runs it as a
service on `PORT` (8080):

| Route | What it does |
|---|---|
| `GET /healthz` | `200` with the corpus size, or `503` when the corpus is missing: answering zero-shot would be a different agent |
| `POST /v1/turn` | One build. Body: `{"request": "...", "context": {...}}` (or `{"task": {...}}` in the CLI's shape), plus `zetta_base_url` and `zetta_token` for the workspace. Answers `{ok, trace_id, result: {pipeline, reply, trajectory, defects}}`; `422` with `error` when the build failed |

- **The workspace is reached through lake.** `zetta_base_url` is lake's
  `/workspaces/<customer>/zetta` path and `zetta_token` the lake key the turn carries, so every read
  goes through lake's allow-lists and audit ([lake#235](https://github.com/oramasearch/lake/issues/235)).
  The reads are the node's real capabilities: `list_datasets`, `get_dataset`, and
  `list_projects` / `list_models` behind `list_pipelines` / `list_explainers`.
- **`AGENT_SERVICE_TOKEN`**, when set, is required as `Authorization: Bearer <token>`: only lake
  calls the service.
- **Every turn is a trace** in `TRACE_DIR`, without the caller's key.
- `PROJECT_MODEL` picks the model (default `gpt-4o`); `OPENAI_API_KEY` is the general Orama key
  the operator mounts.

**Image:** `Dockerfile`, published **private** as `oramasearch/agent-project` (created private
before the first push; an anonymous read answers 404). The corpus is the named build context
`corpus` (`agent-project-corpus/corpus`). Build it on the cluster BuildKit, never with Docker on a
workstation:

```sh
buildctl --addr "kube-pod://<buildkit pod>?context=orama-production-cluster-01&namespace=buildkit" build \
  --frontend dockerfile.v0 --local context=. --local dockerfile=. \
  --local corpus=../agent-project-corpus/corpus --opt context:corpus=local:corpus \
  --opt platform=linux/amd64 \
  --output type=image,name=docker.io/oramasearch/agent-project:<tag>,push=true
```

Tests: `uv run pytest` (`tests/test_service.py`: the turn contract, real read capabilities, the
trace without the key, health without the corpus).

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
