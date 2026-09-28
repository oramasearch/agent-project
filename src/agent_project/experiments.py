"""Read Lake study state and prepare ExperimentsAgent handoffs without writes.

The Project Agent can identify that a methodology trains a model, but it should not turn a
provisional model family or unexamined hyperparameters into a recommendation. Lake owns durable
study state and ExperimentsAgent owns study design. This module reads the former and proposes a
turn to the latter; it never starts a study or promotes a trial.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


READS = {
    "list_experiments": "List studies and tracked runs for one project.",
    "get_experiment_study": "Read one complete study, including trials and promotion state.",
}

PROPOSED_ONLY_ACTIONS = frozenset({
    "agent_turn",
    "start_study",
    "stop_study",
    "stop_experiment_trial",
    "promote_trial",
})


class LakeReader(Protocol):
    """The read half of Lake used by the Project Agent."""

    def read(self, capability: str, **params: Any) -> Any:
        """Call one registered Lake read capability."""


@dataclass(frozen=True)
class Evidence:
    """The exact experiment observations collected for one project."""

    reads: list[dict[str, Any]]
    selected_study: str | None = None
    matching_study: str | None = None

    def public(self) -> dict[str, Any]:
        value: dict[str, Any] = {"reads": self.reads}
        if self.selected_study:
            value["selected_study"] = self.selected_study
        if self.matching_study:
            value["matching_study"] = self.matching_study
        return value


def specs() -> list[dict[str, Any]]:
    """The executable experiment surface: two Lake reads and no action."""
    result = []
    for name, description in READS.items():
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "args": {"type": "object"},
                        "because": {
                            "type": "string",
                            "description": "Quote the prior observation that justifies this read.",
                        },
                    },
                    "required": ["args", "because"],
                },
            },
        })
    return result


def _error(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    if isinstance(error, dict):
        return error
    if value.get("ok") is False:
        return value
    return None


def _read(lake: LakeReader, capability: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        value = lake.read(capability, **params)
    except httpx.HTTPStatusError as error:
        return {
            "capability": capability,
            "args": params,
            "ok": False,
            "error": f"{error.response.status_code} {error.response.text[:200]}",
        }
    except httpx.HTTPError as error:
        return {
            "capability": capability,
            "args": params,
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }
    error = _error(value)
    if error is not None:
        return {
            "capability": capability,
            "args": params,
            "ok": False,
            "error": error,
        }
    return {
        "capability": capability,
        "args": params,
        "ok": True,
        "value": value,
    }


def _turn_values(turn: dict[str, Any]) -> dict[str, Any]:
    task = turn.get("task")
    source = task if isinstance(task, dict) else turn
    context = source.get("context")
    if not isinstance(context, dict):
        context = turn.get("context")
    if not isinstance(context, dict):
        context = {}
    return {
        **source,
        **context,
        "customer_id": turn.get("customer_id") or context.get("customer_id"),
    }


def _registry_entry(result: dict[str, Any], methodology: str) -> dict[str, Any] | None:
    registry = result.get("methodology_registry")
    if not isinstance(registry, dict):
        return None
    entries = registry.get("methodologies")
    if not isinstance(entries, list):
        return None
    return next(
        (
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("id") == methodology
        ),
        None,
    )


def handoff_inputs(
        turn: dict[str, Any],
        result: dict[str, Any]) -> dict[str, Any] | None:
    """Return a model-selection problem, or None for a non-model result."""
    pipeline = result.get("pipeline")
    if not isinstance(pipeline, dict):
        return None
    methodology = pipeline.get("methodology")
    dataset = pipeline.get("dataset")
    if not isinstance(methodology, str) or not methodology:
        return None
    if not isinstance(dataset, str) or not dataset:
        return None
    entry = _registry_entry(result, methodology)
    if entry is None or entry.get("runnable") is not True:
        return None
    model_kinds = entry.get("model_kinds")
    if not isinstance(model_kinds, list) or not model_kinds:
        return None
    choices = [item for item in model_kinds if isinstance(item, str) and item]
    if not choices:
        return None
    values = _turn_values(turn)
    project = values.get("project")
    customer_id = values.get("customer_id")
    if not isinstance(project, str) or not project:
        return None
    if not isinstance(customer_id, str) or not customer_id:
        return None
    return {
        "customer_id": customer_id,
        "project": project,
        "dataset": dataset,
        "methodology": methodology,
        "model_kinds": choices,
        "provisional_model_kind": pipeline.get("model_kind"),
        "label": pipeline.get("label"),
        "features": pipeline.get("features"),
        "metrics": entry.get("metrics") or [],
        "read_context": values,
    }


def _study_id(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if not isinstance(value, dict):
        return None
    identifier = value.get("id") or value.get("study")
    return identifier if isinstance(identifier, str) and identifier else None


def _study_methodology(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    spec = value.get("spec")
    source = spec if isinstance(spec, dict) else value
    methodology = source.get("methodology")
    return methodology if isinstance(methodology, str) and methodology else None


def _listed_studies(read: dict[str, Any]) -> list[dict[str, Any]]:
    if read.get("ok") is not True:
        return []
    value = read.get("value")
    if not isinstance(value, dict):
        return []
    studies = value.get("studies")
    if not isinstance(studies, list):
        return []
    return [study for study in studies if isinstance(study, dict)]


def _requested_study(context: dict[str, Any]) -> str | None:
    for key in ("experiment_study", "study_id", "study"):
        identifier = _study_id(context.get(key))
        if identifier:
            return identifier
    return None


def read_state(
        lake: LakeReader,
        inputs: dict[str, Any]) -> Evidence:
    """List a project's studies and read the selected or matching complete record."""
    common = {
        "customer_id": inputs["customer_id"],
        "project": inputs["project"],
    }
    reads = [_read(lake, "list_experiments", common)]
    studies = _listed_studies(reads[0])
    matching = next(
        (
            study
            for study in studies
            if _study_methodology(study) == inputs["methodology"]
        ),
        None,
    )
    matching_id = _study_id(matching)
    requested_id = _requested_study(inputs["read_context"])
    selected_id = requested_id or matching_id
    if selected_id is None and studies:
        selected_id = _study_id(studies[0])
    if selected_id:
        reads.append(_read(lake, "get_experiment_study", {
            "customer_id": inputs["customer_id"],
            "study": selected_id,
        }))
    return Evidence(
        reads=reads,
        selected_study=selected_id,
        matching_study=matching_id,
    )


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def experiments_agent_proposal(
        inputs: dict[str, Any],
        evidence: Evidence) -> dict[str, Any]:
    """Return the exact read-only agent turn a person may confirm."""
    context = {
        key: inputs[key]
        for key in (
            "project",
            "dataset",
            "methodology",
            "model_kinds",
            "provisional_model_kind",
            "label",
            "features",
            "metrics",
        )
        if inputs.get(key) is not None
    }
    message = (
        f"Design a bounded study for project {inputs['project']} on {inputs['dataset']} using "
        f"methodology {inputs['methodology']}. Compare the applicable model families "
        f"{_json_value(inputs['model_kinds'])} and tune consequential hyperparameters. "
        "Return a study proposal for review; do not start a study or promote a trial."
    )
    return {
        "kind": "proposal",
        "capability": "agent_turn",
        "agent": "agent-experiments",
        "turn_kind": "experiment",
        "confirmation_required": True,
        "request": {
            "agent": "agent-experiments",
            "kind": "experiment",
            "message": message,
            "context": context,
        },
        "inputs": context,
        "evidence": evidence.public(),
    }


def _complete_study(evidence: Evidence) -> dict[str, Any] | None:
    read = next(
        (
            item
            for item in evidence.reads
            if item.get("capability") == "get_experiment_study"
            and item.get("ok") is True
        ),
        None,
    )
    if read is None or not isinstance(read.get("value"), dict):
        return None
    return read["value"]


def study_summary(evidence: Evidence) -> str | None:
    """Explain durable study facts without inventing an analysis result."""
    study = _complete_study(evidence)
    if study is None:
        return None
    identifier = _study_id(study) or evidence.selected_study or "unknown"
    status = str(study.get("status") or "unknown")
    trials = study.get("trials")
    trial_count = len(trials) if isinstance(trials, list) else 0
    summary = f"Study {identifier} is {status} with {trial_count} trial(s)."
    promoted = study.get("promoted_model")
    if isinstance(promoted, str) and promoted:
        summary += f" Its promoted model is {promoted}."
    else:
        summary += " It has no promoted model."
    return summary


def attach(
        turn: dict[str, Any],
        result: dict[str, Any],
        lake: LakeReader) -> dict[str, Any]:
    """Attach study evidence and, when none matches, a confirmation-only handoff."""
    inputs = handoff_inputs(turn, result)
    if inputs is None:
        return result
    evidence = read_state(lake, inputs)
    answer = {
        **result,
        "experiment_evidence": evidence.public(),
    }
    summary = study_summary(evidence)
    reply = str(answer.get("reply") or "").strip()
    if summary:
        answer["experiment_summary"] = summary
        answer["reply"] = f"{reply} {summary}".strip()
    if evidence.matching_study:
        return answer
    proposal = experiments_agent_proposal(inputs, evidence)
    offer = (
        "The current model kind is provisional. Proposed, not started: have ExperimentsAgent "
        "compare model families and hyperparameters after a person confirms this handoff."
    )
    answer["reply"] = f"{str(answer.get('reply') or '').strip()} {offer}".strip()
    answer["experiment_handoff"] = proposal
    return answer
