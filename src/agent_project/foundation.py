"""Propose a solution foundation from live, read-only workspace evidence.

Lake starts these turns with kind ``solution:<project>``.  The selected module, level, dataset,
delivery, and economics are already decisions made by the person.  This module reads the module
and dataset, maps only columns supported by those observations, and leaves absent policy evidence
explicitly missing for confirmation.
"""

from __future__ import annotations

import re
from typing import Any

from agent_project.node import Node


DECISION_OUTPUTS = {
    "M-4": {
        "entity_column": "series",
        "score_column": "point_forecast",
        "output_name": "forecast",
    },
}

ENTITY_NAMES = (
    "entity_id",
    "customer_id",
    "account_id",
    "record_id",
    "series_id",
    "product_id",
    "sku_id",
    "location_id",
    "account",
    "customer",
    "entity",
    "id",
)
LABEL_NAMES = (
    "label",
    "target",
    "outcome",
    "churned",
    "retained",
    "demand",
    "sales",
    "units",
    "quantity",
    "value",
)
TIME_NAMES = (
    "observed_at",
    "observation_time",
    "snapshot_at",
    "snapshot_date",
    "as_of",
    "timestamp",
    "event_time",
    "time",
    "date",
    "period",
    "week",
    "month",
)
NAME_NAMES = (
    "name",
    "customer_name",
    "account_name",
    "product_name",
    "sku_name",
)


def _dataset_parts(dataset: str) -> tuple[str, str]:
    if "." not in dataset:
        return "datasets", dataset
    return tuple(dataset.split(".", 1))


def _solution_entry(value: Any, identifier: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("get_solution returned no solution object")
    nested = value.get("solution")
    entry = nested if isinstance(nested, dict) else value
    if entry.get("id") != identifier:
        raise ValueError(f"get_solution did not return {identifier}")
    return entry


def _selected_methodologies(entry: dict[str, Any], level: int) -> list[dict[str, Any]]:
    selected = []
    for item in entry.get("composition") or []:
        if not isinstance(item, dict):
            continue
        minimum = item.get("minimum_level")
        identifier = item.get("methodology")
        if isinstance(minimum, int) and isinstance(identifier, str) and minimum <= level:
            selected.append(item)
    if not selected:
        raise ValueError(f"get_solution returned no methodologies for level {level}")
    return selected


def _analysis_columns(analysis: Any, dataset: Any) -> list[dict[str, Any]]:
    if not isinstance(analysis, dict):
        raise ValueError("analyze_dataset returned no analysis object")
    raw = analysis.get("columns") or []
    columns = [item for item in raw if isinstance(item, dict) and item.get("name")]
    if columns:
        return columns
    names = dataset.get("columns") if isinstance(dataset, dict) else []
    return [{"name": name, "role_candidates": []} for name in names if isinstance(name, str)]


def _column_names(columns: list[dict[str, Any]]) -> dict[str, str]:
    return {str(column["name"]).lower(): str(column["name"]) for column in columns}


def _role_column(columns: list[dict[str, Any]], roles: set[str],
                 names: tuple[str, ...]) -> str | None:
    for column in columns:
        candidates = column.get("role_candidates") or []
        if any(candidate in roles for candidate in candidates):
            return str(column["name"])
    available = _column_names(columns)
    for name in names:
        if name in available:
            return available[name]
    return None


def _entity_name_priority(name: str) -> int:
    lowered = name.lower()
    if lowered.endswith("_id") or lowered == "id":
        return 3
    if lowered in ENTITY_NAMES:
        return 2
    parts = set(re.split(r"[^a-z0-9]+", lowered))
    if parts.intersection({"customer", "account"}):
        return 1
    return 0


def _has_numeric_evidence(column: dict[str, Any]) -> bool:
    quantiles = column.get("quantiles")
    return isinstance(quantiles, dict) and bool(quantiles)


def _is_measure(column: dict[str, Any]) -> bool:
    semantic_type = str(column.get("semantic_type") or "").lower()
    if semantic_type == "numeric":
        return True
    return _entity_name_priority(str(column["name"])) == 0 and _has_numeric_evidence(column)


def _entity_column(analysis: dict[str, Any], columns: list[dict[str, Any]]) -> str | None:
    key_uniqueness = {}
    keys = analysis.get("candidate_keys") or []
    for key in keys:
        if not isinstance(key, dict):
            continue
        names = key.get("columns") or []
        if len(names) == 1:
            key_uniqueness[str(names[0]).lower()] = float(key.get("uniqueness") or 0)

    candidates = []
    for position, column in enumerate(columns):
        name = str(column["name"])
        roles = set(column.get("role_candidates") or [])
        semantic_type = str(column.get("semantic_type") or "").lower()
        priority = _entity_name_priority(name)
        if roles.intersection({"label", "target", "outcome", "time"}):
            continue
        if not priority and not roles.intersection({"entity", "entity_key"}):
            if semantic_type != "identifier":
                continue
        if _is_measure(column):
            continue
        distinct_count = int(column.get("distinct_count") or 0)
        distinct_ratio = float(column.get("distinct_ratio") or 0)
        repeated = bool(analysis.get("grain") in {"event", "snapshot", "time_series"}
                        and 0 < distinct_ratio < 0.98)
        score = (
            priority,
            bool(roles.intersection({"entity", "entity_key"})),
            repeated,
            distinct_count,
            distinct_ratio,
            key_uniqueness.get(name.lower(), 0),
            -position,
        )
        candidates.append((score, name))
    if not candidates:
        return None
    return max(candidates)[1]


def _latest_complete_period(column: dict[str, Any]) -> Any:
    if "latest_complete_period" in column:
        return column["latest_complete_period"]
    top_values = [
        item
        for item in column.get("top_values") or []
        if isinstance(item, dict) and "value" in item
    ]
    if top_values:
        largest_count = max(int(item.get("count") or 0) for item in top_values)
        complete = [item["value"] for item in top_values
                    if int(item.get("count") or 0) == largest_count]
        if complete:
            numeric = [value for value in complete
                       if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if len(numeric) == len(complete):
                return max(numeric)
            return max(complete, key=str)
    quantiles = column.get("quantiles") or {}
    if isinstance(quantiles, dict) and "max" in quantiles:
        return quantiles["max"]
    return None


def _planned_columns(plan: dict[str, Any], available: set[str],
                     excluded: set[str]) -> list[str]:
    result = []
    entries = plan.get("entries") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidates = entry.get("source_columns") or entry.get("features") or []
        for name in candidates:
            if isinstance(name, str) and name in available and name not in excluded:
                if name not in result:
                    result.append(name)
    for name in plan.get("features") or []:
        if isinstance(name, str) and name in available and name not in excluded:
            if name not in result:
                result.append(name)
    return result


def _feature_columns(columns: list[dict[str, Any]], planned: list[str],
                     excluded: set[str]) -> list[str]:
    features = []
    for column in columns:
        name = str(column["name"])
        if name in excluded:
            continue
        semantic_type = str(column.get("semantic_type") or "").lower()
        supported = semantic_type in {"categorical", "numeric"}
        if semantic_type == "identifier" and _is_measure(column):
            supported = True
        if not semantic_type and name in planned:
            supported = True
        if supported:
            features.append(name)
    return features


def _policy_predicate_names(entry: dict[str, Any]) -> list[str]:
    template = entry.get("policy_template") or entry.get("policy") or {}
    if not isinstance(template, dict):
        return []
    names = []
    for group in ("eligibility_rules", "exclusion_rules"):
        for rule in template.get(group) or []:
            if not isinstance(rule, dict):
                continue
            name = rule.get("name") or rule.get("field")
            if isinstance(name, str) and name and name not in names:
                names.append(name)
    return names


def _predicate_evidence(
    entry: dict[str, Any],
    columns: list[dict[str, Any]],
) -> tuple[dict[str, str], list[str]]:
    available = _column_names(columns)
    predicates = {}
    missing = []
    for predicate in _policy_predicate_names(entry):
        column = available.get(predicate.lower())
        if column is None:
            missing.append(predicate)
        else:
            predicates[predicate] = column
    return predicates, missing


def _methodology_mapping(identifier: str, database: str, table: str,
                         entity: str | None, label: str | None,
                         time: str | None, features: list[str],
                         columns: list[dict[str, Any]]) -> dict[str, Any]:
    if identifier == "M-2":
        mapping: dict[str, Any] = {"source": database, "table": table}
        if entity:
            mapping["record_id"] = entity
        name = _role_column(columns, {"name"}, NAME_NAMES)
        if name:
            mapping["name"] = name
        return mapping
    if identifier == "M-4":
        mapping = {}
        if time:
            mapping["time_column"] = time
        if label:
            mapping["target_column"] = label
        if entity:
            mapping["series_column"] = entity
        return mapping
    mapping = {}
    if entity:
        mapping["entity_column"] = entity
    if label:
        mapping["label_column"] = label
    if time:
        mapping["time_column"] = time
    mapping["feature_columns"] = features
    return mapping


def _target_table(project: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", project).strip("_").lower()
    return f"{name or 'solution'}_snapshot"


def _mapping_explanations(sources: dict[str, Any], selected: list[dict[str, Any]]) -> list[str]:
    required = {
        "M-1": ("entity_column", "label_column", "feature_columns"),
        "M-2": ("record_id", "name"),
        "M-3": ("entity_column", "feature_columns"),
        "M-4": ("time_column", "target_column", "series_column"),
    }
    explanations = []
    for item in selected:
        identifier = item["methodology"]
        mapping = sources.get(identifier) or {}
        for role in required.get(identifier, ()):
            if role not in mapping or mapping[role] in (None, "", []):
                explanations.append(
                    f"No observed column mapping supports {identifier}.{role}."
                )
    return explanations


def _required_context(turn: dict[str, Any]) -> tuple[dict[str, Any], str, str, int, str]:
    context = turn.get("context")
    if not isinstance(context, dict):
        raise ValueError("a solution proposal needs its start context")
    kind = str(turn.get("kind") or "")
    project = str(context.get("project") or kind.removeprefix("solution:"))
    solution = context.get("solution")
    level = context.get("level")
    dataset = context.get("dataset")
    if not project or not isinstance(solution, str) or not solution:
        raise ValueError("a solution proposal needs its project and solution")
    if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 4:
        raise ValueError("a solution proposal needs a level from 1 through 4")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("a solution proposal needs its selected dataset")
    return context, project, solution, level, dataset


def propose(turn: dict[str, Any], node: Node) -> dict[str, Any]:
    """Read the selected module and dataset, then return Lake's foundation proposal shape."""
    context, project, solution, level, dataset_path = _required_context(turn)
    database, table = _dataset_parts(dataset_path)
    trajectory = []

    solution_value = node.read("get_solution", solution=solution)
    trajectory.append({"call": "get_solution", "args": {"solution": solution},
                       "value": solution_value})
    entry = _solution_entry(solution_value, solution)
    selected = _selected_methodologies(entry, level)

    analysis = node.read("analyze_dataset", dataset=dataset_path)
    trajectory.append({"call": "analyze_dataset", "args": {"dataset": dataset_path},
                       "value": analysis})
    dataset = node.read("get_dataset", dataset=dataset_path)
    trajectory.append({"call": "get_dataset", "args": {"dataset": dataset_path},
                       "value": dataset})
    columns = _analysis_columns(analysis, dataset)
    available = {str(column["name"]) for column in columns}
    entity = _entity_column(analysis, columns)
    label = _role_column(columns, {"label", "target", "outcome"}, LABEL_NAMES)
    time = _role_column(columns, {"time"}, TIME_NAMES)
    time_detail = next((column for column in columns if column.get("name") == time), {})
    as_of = context.get("as_of")
    if as_of is None:
        as_of = _latest_complete_period(time_detail)

    plans = {}
    for item in selected:
        identifier = item["methodology"]
        args: dict[str, Any] = {"dataset": dataset_path, "methodology": identifier}
        if label:
            args["label"] = label
        if time:
            args["time_column"] = time
        if as_of is not None:
            args["as_of"] = as_of
        plan = node.read("plan_features", **args)
        plans[identifier] = plan if isinstance(plan, dict) else {}
        trajectory.append({"call": "plan_features", "args": args, "value": plan})

    primary_item = next((item for item in selected if item.get("role") == "primary"), selected[0])
    primary = primary_item["methodology"]
    predicates, missing_predicates = _predicate_evidence(entry, columns)
    excluded = {name for name in (entity, label, time, *predicates.values()) if name}
    primary_plan = _planned_columns(plans.get(primary, {}), available, excluded)
    features = _feature_columns(columns, primary_plan, excluded)
    sources: dict[str, Any] = {"source": database, "table": table}
    for item in selected:
        identifier = item["methodology"]
        method_plan = _planned_columns(plans.get(identifier, {}), available, excluded)
        method_features = _feature_columns(columns, method_plan, excluded)
        sources[identifier] = _methodology_mapping(
            identifier,
            database,
            table,
            entity,
            label,
            time,
            method_features,
            columns,
        )

    snapshot: dict[str, Any] = {
        "database": database,
        "source_table": table,
        "target_table": _target_table(project),
    }
    if entity:
        snapshot["entity_column"] = entity
    if time:
        snapshot["time_column"] = time
    if as_of is not None:
        snapshot["as_of"] = as_of
    dataset_roles: dict[str, Any] = {"feature_columns": features}
    if entity:
        dataset_roles["entity_column"] = entity
    if label:
        dataset_roles["label_column"] = label

    foundation: dict[str, Any] = {
        "sources": sources,
        "snapshot": snapshot,
        "dataset": dataset_roles,
        "delivery": context.get("delivery") or {},
        "roi_model": context.get("roi_model") or {},
        "predicates": predicates,
        "missing_predicates": missing_predicates,
    }
    if primary != "M-1" and primary in DECISION_OUTPUTS:
        foundation["decision_source"] = {"methodology": primary, **DECISION_OUTPUTS[primary]}

    explanations = _mapping_explanations(sources, selected)
    for role, value in (("entity", entity), ("time", time), ("label", label)):
        if value is None:
            explanations.append(f"No observed column supports the {role} role.")
    if as_of is None:
        explanations.append("The dataset analysis supplied no latest complete period for as_of.")
    if primary != "M-1" and "decision_source" not in foundation:
        explanations.append(f"No observed output contract supports a decision source for {primary}.")
    if not features:
        explanations.append("The primary methodology plan supplied no feature columns.")
    reply = "Proposed a read-only solution foundation from the workspace observations."
    if explanations:
        reply += " " + " ".join(explanations)
    return {
        "kind": "solution_foundation",
        "foundation": foundation,
        "reply": reply,
        "trajectory": trajectory,
        "read_only": True,
    }
