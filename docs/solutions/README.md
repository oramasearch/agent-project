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

An engine that answers `unknown_capability` to `list_solutions` predates the solution registry.
That one response selects the existing methodology-only path. Other registry failures are not
treated as an empty catalogue and do not enable a compatibility fallback.
