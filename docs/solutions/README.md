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
marked for owner sign-off. A requested policy change reads the current policy and returns a
before/after proposal; it neither rebuilds the solution nor applies the policy.

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
