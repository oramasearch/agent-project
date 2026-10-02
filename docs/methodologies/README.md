# Methodology-aware pipeline design

The Project Agent treats Zetta's methodology registry as workspace state, not prompt knowledge.
Build and edit turns both read `list_methodologies` before asking the model for a decision. The
observed response is retained in the turn trajectory or proposal reads, so a reviewer can see the
availability facts used for the decision.

## Build flow

For a runnable method, the turn follows this order:

1. `list_methodologies` supplies runnable flags, model kinds, required inputs and tracking issues.
2. `list_datasets` / `get_dataset` identify the real source.
3. `analyze_dataset` reports semantic types, grain, keys, joins and leakage suspects.
4. `plan_features {dataset, methodology}` returns concrete features or a transform-first result.
5. `compile_intent {project, methodology}` (Zetta compiles the project's captured intent and accepts no `intent` argument) returns the methodology-owned graph.
6. The reviewed design names `dataset`, `methodology`, `model_kind`, and features or the required
   transform. A label is required only when the registry marks a label, target or outcome input as
   required.

The validator checks that analysis, planning and compilation occurred, that planning and
compilation used the design's methodology, and that the model kind belongs to the live registry
entry. If the model emits a pipeline for an absent or non-runnable method, the runtime replaces it
with a decline derived from the registry.

## Unsupported and older engines

A non-runnable registry entry produces no pipeline. The result names the method, why it cannot run,
and the entry's `issue`. There is no fallback to a different method.

`unknown_capability` is the one compatibility case. It identifies an engine from before the
registry and creates a turn-local view containing only runnable M-1 churn / GBDT classification.
Any other request is declined with the old-engine limitation. Other failures to read the registry
do not enable a fallback.

## Editing a pipeline

Edit mode keeps the review-before-apply contract. It reads the registry and data, asks
`plan_features`, and calls `compile_intent` for the requested method. The proposed graph operations
must reproduce that compiled shape, including methodology-specific steps and the model step's
`model_kind`. Operations are simulated in order against a copy of the current graph and any
operation that would fail is reported. The agent never claims that the proposal was applied.

The faked-workspace tests cover runnable M-3 segmentation, runnable M-4 weekly forecasting, an
unsupported method with its issue, the old-engine M-1-only response, and an edit that replaces a
churn graph with segmentation's compiled shape.
