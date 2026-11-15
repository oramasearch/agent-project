# Solution-aware operation

The Project Agent discovers solution modules from each workspace. At the beginning of a solution
turn it calls `list_solutions` and retains the complete response with the trajectory. It does not
carry a local S-1 through S-12 or B-1 through B-8 table, because runnable stages, missing
foundation components, levels, data roles and promotion conditions belong to the engine.

`get_solution` reads the chosen entry again before the agent builds it. The workspace operations
available to a solution turn are:

- reads: `get_policy`, `get_experiment`, `list_ledger`, `get_monitor`, and `get_roi_readout`;
- actions: `create_solution`, `run_solution`, `decide`, and `approve_decision`.

The tool surface deliberately omits `create_policy` and `update_policy`. A policy definition or
change is a proposal for an owner to approve, never an unreviewed mutation by this agent.

## Start this solution foundation

Lake's **Start this solution** flow sends a bus turn whose kind is `solution:<project>`. This is
separate from the build-and-operate flow below: the task credential is read-only, the turn does not
use `create_solution`, and the returned `result.foundation` remains a proposal until the person
uses lake's `confirm_solution` action.

The turn reads the selected module with `get_solution`, then reads the selected table with
`analyze_dataset` and `get_dataset`. It selects every methodology whose `minimum_level` is at or
below the requested level and calls `plan_features` for each one. The proposal contains:

- `sources`: the person's database and table plus a role mapping for every selected methodology;
- `snapshot`: source and target tables, entity and time columns, and `as_of`;
- `dataset`: the observed entity and label plus the remaining numeric and categorical feature
  columns after structural and policy columns are removed;
- `decision_source`: the selected primary methodology's documented output mapping when it is not
  M-1 (M-4 forecasts use output `forecast`, entity `series`, and score `point_forecast`);
- `predicates`: policy predicate names mapped only to same-named observed columns;
- `missing_predicates`: every policy predicate for which the table supplied no column evidence;
- the person's unchanged `delivery` and `roi_model` values.

For a sampled time column, `as_of` uses the latest value among the most populated observed periods;
a numeric analysis with no populated-period evidence uses its observed maximum. If analysis cannot
support an entity, time, label, feature, methodology role, cutoff, or non-M-1 output contract, the
agent does not choose one: the reply explains what remains unmapped. In particular, absent S-1
`in_collections`, `open_complaint`, and `vulnerable` flags remain missing rather than becoming
`false`.

Entity selection prefers observed entity-key or identifier columns whose names identify customers,
accounts, or ids, with high cardinality and repeated values across observed periods. Numeric
measures are not accepted as entities even when their sampled values are unique. Feature selection
excludes the entity, label, time, and every observed policy predicate; it retains the other observed
numeric and categorical columns. M-2 uses the selected entity as `record_id` and a name-like column
as `name`. When no name-like column was observed, the proposal leaves `name` unset and explains the
missing `M-2.name` mapping for review.

## Which turns operate a solution

Only a request that **names a solution** (`solution` on the task or the turn's `context`) is a
solution turn. Any other request, such as "Build a pipeline that predicts churn", is a pipeline
design on the methodology path (`docs/methodologies/README.md`), whose answer is the
`result.pipeline` Lake applies to a project.

**A Lake turn never builds.** Lake's turn key for the Project Agent grants no solution action
(lake `docs/api/agent-launches.md`): `create_solution` composes a pipeline, policy, protected
experiment, monitor and ROI definition, and `run_solution` delivers governed actions, so Lake runs
them only when the person confirms (`start_solution`, then `confirm_solution`). Every turn the
service takes (`service.run_turn`) therefore runs solution mode with `act` false: the action tools
are not offered, a call to one is rejected before it reaches the workspace, and the turn reads the
entry with `get_solution` and answers `kind: solution_proposal`:

- `solution`: the id, project, an integer level from the live entry, and the `sources` to build
  from, with `production_ready` and `missing` from the entry;
- the policy proposal and label definitions below, all for owner sign-off;
- a reply ending *Proposed, not built*, and `applied: false`.

A proposal for an entry `get_solution` did not read, a sketch, or a level absent from the entry is
declined. The build-and-operate flow below runs only from the command line (`agent-project`, `act`
true), against a workspace and a key of your own.

To verify: `uv run pytest tests/test_solutions.py -k lake_turn`.

## Build and operate flow

For a selected module, the turn follows this order:

1. `get_solution` reads the current detailed entry. The runtime checks the chosen level and the
   module's status against this observation.
2. `create_solution {project, solution, level, sources}` builds the composed graph. The sources
   must name a real `source` and `table`.
3. `run_solution {project, graph}` may use only the graph and project returned by the successful
   create call.
4. `get_policy`, `get_experiment`, `list_ledger`, `get_monitor`, and `get_roi_readout` collect the
   deployed evidence used in the explanation. `unknown_capability` from any of these optional
   reads is reported as “not deployed yet.”

The returned policy proposal uses the live entry's policy template, autonomy defaults and
guardrails. Label definitions use its data-role descriptions and measurement design. Both are
marked for owner sign-off. A requested policy change takes its `before` value directly from the
`get_policy` observation and returns a proposed replacement; it neither rebuilds the solution nor
applies the policy.

The result pairs the exact `get_roi_readout` value with the model's explanation. Its status is
`available`, `not_deployed_yet`, or `unavailable`, so callers can distinguish measured evidence
from an operation the workspace does not provide.

`approve_decision` is separate from policy editing: it releases an A1 decision only when the turn
contains matching explicit owner approval for that decision id.

## Honest readiness

A non-sketch module can build and run a preview while `runnable: false` names missing production
components. The result sets `production_ready: false` and repeats every missing component's issue.
It does not describe that preview as deployed production operation. A `status: sketch` entry is
read with `get_solution` and refused with its live promotion condition.

An engine that answers `unknown_capability` to `list_solutions` predates the solution registry.
That one response selects the existing methodology-only path. Other registry failures are not
treated as an empty catalogue and do not enable a compatibility fallback.
