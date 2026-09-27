"""The tool surface, and the rule that makes a trajectory a trajectory.

This agent does not answer questions about pipelines — it builds them, by calling tools and
deciding the next call from what came back. The corpus calls that `trajectory`, and the property
that distinguishes it from a lookup is mechanical:

**every call after the first names the observation that justified it.**

A call justified by the user's ask is a scripted step, and a scripted trajectory is a lookup with
extra steps. So `because` is a required argument on every call but the first, and the loop rejects
a call that omits it rather than quietly accepting a plausible-looking sequence.
"""

from __future__ import annotations

from typing import Any

# Named from the calls the corpus actually contains, so the runtime surface and the training
# surface are the same set. An agent trained on `temporal_split` and deployed against a runtime
# that never offers it would fail in a way no eval would show.
READS = {
    "list_datasets": "What data is captured. Takes no arguments. This is where a trajectory starts.",
    "get_dataset": "The columns, types and null fractions of one dataset. Args: dataset.",
    "list_pipelines": "The pipelines that already exist. Takes no arguments.",
    "list_explainers": "The explainers available to attach. Takes no arguments.",
}
BUILDS = {
    "clean": "Drop or repair rows. Args: drop_null, as_of.",
    "merge": "Join two datasets. Args: left, right, on.",
    "aggregate": "Roll rows up to an entity. Args: by, measures.",
    "temporal_split": "Split by time rather than at random — for forecasting. Args: as_of, horizon.",
    "model": "Fit a supervised model. Args: label, features, model_kind, calibrated, holdout.",
    "forecast": "Fit a forecaster. Args: series, horizon, backtest.",
    "uplift_model": "Fit an uplift model. Args: treatment, outcome, incrementality.",
    "detect_anomalies": "Fit an unsupervised detector. Args: signal, threshold_governed.",
    "recommendation": "Build a retrieval index. Args: index, grounded_in_data.",
    "add_explanation": "Attach an explainer. Args: explainer.",
    "serve": "Expose the result. Args: name.",
    "update_pipeline": "Describe a change to an existing pipeline (recorded in the pipeline you "
                       "return; nothing is saved). Args: pipeline, change.",
}


def specs() -> list[dict[str, Any]]:
    """The tools as the OpenAI SDK wants them.

    `because` is declared required on every tool including the first call's, and enforced by the
    loop rather than the schema — the first call has no prior observation to name, and a schema
    cannot express "required except at position zero".
    """
    tools = []
    for name, description in {**READS, **BUILDS}.items():
        tools.append({"type": "function", "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {"type": "object",
                             "description": "The call's arguments, as named in the description."},
                    "because": {"type": "string",
                                "description": "The PREVIOUS OBSERVATION that justifies this call. "
                                               "Quote what you saw, not what the user asked for. "
                                               "Omit only on the very first call."},
                },
                "required": ["args"],
            }}})
    return tools


def undisciplined(trajectory: list[dict[str, Any]],
                  before: dict[str, Any] | None = None) -> list[str]:
    """Where the trajectory stopped reading and started reciting.

    `before` — the pipeline being changed — counts as observed. It is state the agent was HANDED,
    so "the model is not calibrated" is reading it, not reciting the ask. Excluding it would fail
    every honest mutation for citing the one thing a mutation is about.
    """
    problems: list[str] = []
    if not trajectory:
        return ["the trajectory is empty — nothing was read before building"]

    first = trajectory[0]["call"]
    if first not in READS:
        problems.append(f"starts with {first!r}, which builds something before reading anything")

    for index, step in enumerate(trajectory[1:], start=1):
        because = (step.get("because") or "").strip()
        if not because:
            problems.append(f"step {index} ({step['call']}) names no observation that justified it")
            continue
        # It must cite something that was actually seen. Naming a tool that has not run yet, or
        # citing the ask, is the scripted failure wearing a justification.
        seen = {prior["call"] for prior in trajectory[:index]}
        observations = " ".join(
            [str(prior.get("observation", "")) for prior in trajectory[:index]]
            + [f"{key} {value}" for key, value in (before or {}).items()])
        if not any(call in because for call in seen) and not _overlaps(because, observations):
            problems.append(f"step {index} ({step['call']}) cites {because!r}, which does not "
                            "refer to anything already observed")
    return problems


def _overlaps(because: str, observations: str) -> bool:
    """Does the justification quote the observations at all?

    Deliberately crude — a shared content word is enough. The check exists to catch a justification
    invented wholesale ("the user wants churn prediction"), not to grade paraphrase quality, and a
    strict version would reject honest rewording.
    """
    stop = {"the", "a", "an", "is", "are", "was", "has", "have", "that", "this", "it", "and", "to",
            "of", "in", "for", "on", "with", "so", "which", "showed", "shows", "reported"}
    words = {word.strip(".,;:()[]'\"`").lower() for word in because.split()}
    seen = {word.strip(".,;:()[]'\"`").lower() for word in observations.split()}
    return bool((words - stop) & (seen - stop))
