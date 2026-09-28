# ApplicationAgent handoff

The Project Agent can identify project output that benefits from a UI and prepare a handoff to
ApplicationAgent. The handoff is an unapplied `agent_turn` proposal with `agent:
agent-application`, turn kind `app`, and context containing the exact result tables, model metadata,
result meaning, and project. A person confirms that proposal before Lake starts ApplicationAgent.
ApplicationAgent then returns its own file and App proposal for a separate review. The Project
Agent never writes an App or deploys one.

When the turn names an existing `app_id`, that identifier is included in the proposed context and
the request asks ApplicationAgent to change that App. Without one, it asks ApplicationAgent to
build a new data app.

The runtime offers this for result kinds `scored_list`, `ranked_list`, `scores`, `forecast`,
`forecast_table`, `segment_table`, and `segments` (hyphens and spaces normalize to underscores).
The turn must supply a project, one or more result tables, and model metadata. Training metrics or
a pipeline design with no materialized result do not trigger an App offer.

## Lake evidence

Apps and builds are Lake resources, not workspace Zetta resources. The Project Agent uses the
turn's short-lived `lake_url` and `lake_token` for these reads:

- `list_apps` and `get_app` read desired and observed App state;
- `receive_app_events` reads retained rollout events from a one-based cursor;
- `list_builds` and `get_build` read build state;
- `receive_build_events` reads retained build progress from a one-based cursor.

Every handoff reads both lists. When the turn identifies an App or build, it also reads that row
and its retained events. Read failures remain in the proposal evidence; they are not replaced with
invented state.

## Confirmation boundary

The executable App tool surface contains only the six reads above. `agent_turn`, `create_app`,
`deploy_app`, `build_app`, `rollback_app`, `scale_app`, lifecycle operations, environment changes,
binding changes, authentication changes, and deletion are proposal-only operations. In particular,
the Project Agent does not receive an App action method and cannot deploy itself.

After a person confirms the handoff, Lake sends the initial read-only `app` task to
`agent-application`. Only a later ApplicationAgent task whose context contains the exact reviewed
`confirmed_proposal` receives repository writes, `create_app`, and `deploy_app`.

The trace clears both `lake_token` and `zetta_token`; neither delegated credential is retained.
