# ExperimentsAgent handoff

The Project Agent treats a methodology's model kind as a provisional runnable baseline, not a
claim that it is the best model. When a runnable methodology has model kinds, the service reads
that project's study state and either explains the matching study or offers ExperimentsAgent a
model-family and hyperparameter study.

## Lake evidence

The Project Agent uses only two registered Lake read capabilities:

- `list_experiments {customer_id, project}` finds studies and standalone tracked runs for the
  project;
- `get_experiment_study {customer_id, study}` reads the selected study's objective, budgets,
  trial state, cost, and promoted model.

If the turn names `experiment_study`, `study_id`, or `study`, that study is read. Otherwise the
service prefers a study for the pipeline's methodology, then the first project study. A matching
study is summarized from the complete record and suppresses a duplicate proposal. Read failures
remain in `experiment_evidence`; they are not replaced with invented results.

## Delegation and confirmation

With no matching study, `experiment_handoff` is an unapplied `agent_turn` proposal for library key
`agent-experiments`, turn kind `experiment`. Its context carries the exact project, dataset,
methodology, available model kinds, provisional model kind, label, features, and registry metrics.
The request asks ExperimentsAgent to return a bounded study proposal and explicitly forbids
starting a study or promoting a trial.

A person confirms the handoff before Lake starts ExperimentsAgent. That agent may read methodology
and dataset facts and return its own study proposal. Only a later turn carrying the exact reviewed
proposal may call `start_study`. Likewise, a completed trial is promoted only after a separate
reviewed promotion confirmation.

The Project Agent exposes no experiment action. `start_study`, `stop_study`,
`stop_experiment_trial`, and `promote_trial` are proposal-only here. The unit contract lists all
eleven Lake experiment MCP tools and asserts that the two capabilities this agent calls are in
that registered `tools/list` surface.
