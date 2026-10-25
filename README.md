# agent-project

Builds and operates catalogue Solutions, and builds or changes Pipelines, by **calling tools**.
It takes a profile and an intent — the shape `agent-autosuggest` emits — and walks a trajectory
against a live node.

**Current release: [v0.2.0](docs/releases/v0.2.0.md)** (all releases: [`docs/releases/`](docs/releases/README.md)).

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

`task.json` carries a `profile`, an `intent`, optionally a `project`, requested `solution`, `level`,
`sources`, or requested `methodology`, and — for a change to an existing pipeline — a `before`.

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
| `GET /healthz` | `200` with the package `version` and the corpus size, or `503` when the corpus is missing: answering zero-shot would be a different agent |
| `POST /v1/turn` | One solution operation, methodology build or edit proposal. Body: `{"request": "...", "context": {...}}` (or `{"task": {...}}` in the CLI's shape), plus `zetta_base_url` and `zetta_token` for the workspace. Answers `{ok, trace_id, result}`; `422` with `error` when the turn failed |

- **Turns arrive on lake's agent bus** ([lake#269](https://github.com/oramasearch/lake/issues/269),
  lake `docs/api/agent-bus.md`). `lake_bus.py` long-polls `receive_agent_tasks` through lake,
  claims each task (its body arrives with the task's own credential), runs it as a `/v1/turn`
  would, and reports with `publish_agent_event`: `progress`, then the answer or `failed`. It starts
  when the operator sets `LAKE_URL`, `AGENT_BUS_TOKEN` (this agent's own bus credential) and
  `AGENT_BUS_AGENT`. No request to lake ever waits on the agent.
- **Start this solution is a read-only proposal.** A bus turn whose kind is
  `solution:<project>` reads `get_solution`, `analyze_dataset`, `get_dataset`, and
  `plan_features` for every methodology composed at the chosen level. It returns
  `result.foundation` for the person to review; only lake's later `confirm_solution` call may
  create the solution. Missing policy predicate columns stay in `missing_predicates`.
- **Project results can be handed to ApplicationAgent.** A scored list, forecast, or segment table
  with its result tables and model metadata gains an `app_handoff` proposal. The agent reads Lake's
  App and build state, passes those exact outputs to an `agent-application` turn of kind `app`, and
  waits for a person to confirm the handoff. It never calls App or build mutations and never
  deploys an App itself.
- **Model selection can be handed to ExperimentsAgent.** A methodology pipeline with a trainable
  model gains a read-only study check. The agent uses Lake's registered `list_experiments` and
  `get_experiment_study` capabilities to explain an existing matching study; otherwise it returns
  an unapplied `agent-experiments` turn proposal over the project's dataset, methodology and model
  choices. A person confirms that handoff, the later study start, and any later trial promotion.
- **The workspace is reached through lake.** `zetta_base_url` is lake's
  `/workspaces/<customer>/zetta` path and `zetta_token` the lake key the turn carries, so every read
  goes through lake's allow-lists and audit ([lake#235](https://github.com/oramasearch/lake/issues/235)).
  A turn first reads `list_solutions`. A request that names a solution operates it; any other
  request, and every request on an older engine (whose `unknown_capability` response marks it),
  takes the methodology path. Solution reads include `get_solution`, `get_policy`,
  `get_experiment`, `list_ledger`, `get_monitor`, and `get_roi_readout`; solution actions include
  `create_solution`, `run_solution`, `decide`, and approval of an explicitly authorized A1
  decision, offered only from the command line: a Lake turn's key grants none of them, so a turn
  proposes the solution instead (lake#796). Methodology reads are `list_methodologies`, `list_datasets`,
  `get_dataset`, `analyze_dataset`, `plan_features`, and `list_projects` / `list_models` behind
  `list_pipelines` / `list_explainers`. `compile_intent` is an action; it returns the selected
  methodology's engine-owned pipeline shape. Only the arguments Zetta's compile_intent accepts
  (`project`, `methodology`, `solution`, `hints`, `name`, `input_id`) are sent; the model's own
  goal text is not, because the compiler reads the project's captured intent (#20).
- **`AGENT_SERVICE_TOKEN`** is required as `Authorization: Bearer <token>`, compared in constant
  time (`401` otherwise): only lake calls the service. With it unset or empty the route fails
  closed and refuses every turn with `503`.
- **Every turn is a trace** in `TRACE_DIR`, without the caller's key.
- `PROJECT_MODEL` picks the model (default `gpt-4o`). **The model is reached through lake**
  (lake#248, lake `docs/api/model-gateway.md`): every call goes to `<LAKE_URL>/openai/v1` with the
  turn's own `lake_token` as the API key (`model.py`), so lake counts its tokens in the turn's
  workspace and enforces the account's model-token limit. The service holds no `OPENAI_API_KEY`;
  a turn without a lake credential is refused. Only the command line (`Run`, above), which runs for
  no workspace, calls OpenAI directly with your own key.
- **A turn's credential goes only to Lake at this service's own `LAKE_URL`** (#30). The model
  address is always `LAKE_URL`; a turn's `lake_url` is never used for it. The workspace address
  (`zetta_base_url`, Lake's `<LAKE_URL>/w/<workspace>/workspaces/<customer>/zetta`) and the Lake
  address for App and Experiments reads (`lake_url`) are accepted only on `LAKE_URL`'s origin
  (scheme, host and port) under its path, with no user info, query or fragment; any other address
  refuses the turn before a request is made. With `LAKE_URL` unset, every turn that would send its
  credential is refused. A turn that names no `zetta_base_url` uses the operator's own
  `ZETTA_BASE_URL`.

**The solutions catalogue** in the image is built at image-build time by zetta's own
`scripts/build_catalog.py` from the solutions-designs plan, and passed as the named build context
`catalog` (`catalog.json` and `catalog.source.json`, which records the zetta and solutions-designs
commits). `ZETTA_CATALOG` points at it; without it the agent builds the catalogue from a bridge
checkout, as before.

**Image:** `Dockerfile`, published **private** as `oramasearch/agent-project` (created private
before the first push; an anonymous read answers 404). It needs two named build contexts: `catalog`
(above) and `corpus` (`agent-project-corpus/corpus`). Build it on the cluster BuildKit, never with
Docker on a workstation:

```sh
buildctl --addr "kube-pod://<buildkit pod>?context=orama-production-cluster-01&namespace=buildkit" build \
  --frontend dockerfile.v0 --local context=. --local dockerfile=. \
  --local catalog=<dir with catalog.json and catalog.source.json> --opt context:catalog=local:catalog \
  --local corpus=../agent-project-corpus/corpus --opt context:corpus=local:corpus \
  --opt platform=linux/amd64 \
  --output type=image,name=docker.io/oramasearch/agent-project:<tag>,push=true
```

Tests: `uv run pytest` (`tests/test_service.py`: the turn contract, real read capabilities, the
trace without the key, health without the corpus; `tests/test_lake_bus.py`: receive, claim, run
and publish on the bus; `tests/test_methodologies.py`: runnable, unsupported and old-engine
methodology behavior, feature planning and methodology-changing edits; `tests/test_solutions.py`:
runnable and preview modules, sketch refusal, old-engine discovery, policy approval boundaries and
measured readout operation; `tests/test_foundations.py`: S-1 predicate evidence and S-5 forecast
decision-source proposals; `tests/test_apps.py`: Lake App/build reads, result handoff, credential
scrubbing, and confirmation-only writes; `tests/test_experiments.py`: registered Lake study reads,
existing-study explanations, model-selection handoffs, and confirmation boundaries;
`tests/test_release.py`: the version in `pyproject.toml` has release notes in `docs/releases/`).

## Solutions come from the workspace

A turn that names a solution chooses only from the live `list_solutions` response and then
reads the selected module with `get_solution`; any other request is a pipeline design
(`docs/methodologies/README.md`). **A Lake turn proposes a solution and never builds it**: its key
grants no solution action, so the action tools are withheld and the answer is a
`solution_proposal` the person confirms with Lake's `confirm_solution` (lake#796,
`docs/solutions/README.md`). From the command line, against your own workspace key, the agent
builds the composed graph with `create_solution`, runs exactly the graph that action returned, and reads policy, holdout, ledger, monitor and measured
readout state before explaining the result. Optional operations that answer `unknown_capability`
are reported as not deployed yet.

A module can build and run as a preview while its entry says `runnable: false` and lists missing
production components. The answer names those issues and does not describe the preview as
production-ready. Band-B entries with `status: sketch` are refused with their live promotion
condition.

The live entry also supplies the proposed policy template, autonomy defaults, guardrails, data-role
definitions and measurement definition. These are returned for owner sign-off. Policy changes are
before/after proposals only: the agent exposes no policy-create or policy-update tool. An A1
decision can be approved only when the turn carries matching explicit owner approval.

## Methodologies come from the workspace

Every methodology build and edit turn reads `list_methodologies`. The agent chooses only an entry
whose live response says `runnable: true`; its design names that `methodology` id and one of the
entry's `model_kinds`. It does not carry a local copy of the 21-method portfolio. If the requested
method is not runnable, the answer declines it and names the registry's tracking issue instead of
substituting churn or a generic graph.

After choosing a method, the agent reads `analyze_dataset`, then `plan_features`. The design states
the planned features or the transform that needs approval first, such as aggregating event data to
an entity snapshot. It calls `compile_intent` with the same methodology and uses the returned graph
as the required pipeline shape. Edit mode compares the existing graph with that compiled shape and
returns checked `edit_pipeline` operations for a person to confirm.

The design's model kind is a provisional runnable baseline. For service turns that identify a
project and customer, the runtime reads that project's studies. It explains a matching study from
the complete durable record, or proposes an ExperimentsAgent turn to compare the methodology's
model families and consequential hyperparameters. Neither the handoff nor the study starts without
confirmation, and promotion requires its own later confirmation. See
[`docs/experiments/README.md`](docs/experiments/README.md).

An older workspace may answer `unknown_capability` to `list_methodologies`. That compatibility
response means only M-1 churn / GBDT classification runs. Requests for other methods are declined
with that engine limitation stated explicitly.

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

### Edit mode: a proposal, never a save (agent-project#4)

A turn whose `context` carries the current `pipeline` (nodes with attributes, edges) with its
`project` and `graph` is a **change request** (`edit.py`). The agent reads the workspace
(`list_datasets`, `get_dataset`), then returns:

```json
{"kind": "proposal", "reply": "Proposed, not applied: …",
 "proposal": {"project": "…", "graph": "…", "summary": "…",
              "edits": [{"op": "remove_step", "step": "load_activity"}, …],
              "steps": ["Remove step load_activity and its edges", …],
              "run": {"dataset": "datasets.ux_review_churn", "holdout": 0.2},
              "problems": [], "after": {"nodes": […], "edges": […]}}}
```

`edits` are lake's `edit_pipeline` operations (`add_step`, `rename_step`, `kind_step`,
`remove_step`, `add_edge`, `remove_edge`, `set_attribute`, `rename`), each checked in order against
the graph by `simulate`; `problems` names every one that would not apply. The reply is written from
the checked edits, never by the model, so it cannot claim a change was made: the agent's lake key is
read-only and lake applies the edits only when the person confirms on the pipeline page (lake#249).
In build mode the pipeline returned is a design too (`applied: false`). It always names `dataset`
(the one table it trains on, `<database>.<table>`), `methodology`, `model_kind`, and its features or
transform-first step (prompt v3; `normalize` takes `dataset` from `training_sources` when the model
put it there). It names `label` when the registry says the method requires a label, target or
outcome. lake applies the reviewed design to a project made from an Opportunity with those fields.

[#989]: https://github.com/oramasearch/zetta/issues/989
