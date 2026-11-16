# SPDX-License-Identifier: MIT

"""Check the answer before it reaches a caller.

Two failures matter and neither shows up as malformed JSON. A citation to a solution that does not
exist reads as grounded and is not. A pipeline shape that contradicts its own methodology — a
calibrated forecast — is a plausible-looking answer that would produce a leaking pipeline
downstream.

The methodology rules mirror `agent-project-corpus`, which enforces the same shapes when generating
training data. Both point at one fact: what each methodology requires and what it must not have.
"""

from __future__ import annotations

from typing import Any

from agent_project import catalog

REQUIRED: dict[str, set[str]] = {
    "gbdt": {"point_in_time", "leakage_free", "calibrated", "holdout"},
    "classifier": {"point_in_time", "leakage_free", "calibrated", "holdout"},
    "forecast": {"temporal_validation", "backtest", "holdout"},
    "anomaly": {"threshold_governed"},
    "retrieval": {"index", "grounded_in_data"},
    "uplift": {"incrementality", "holdout"},
}
FORBIDDEN: dict[str, set[str]] = {
    # No probability to calibrate, and a calibrator fit on content-held-out rows re-introduces the
    # leak temporal validation exists to prevent.
    "forecast": {"calibrated"},
    "anomaly": {"calibrated", "leakage_free"},
    "retrieval": {"calibrated", "leakage_free", "holdout"},
}


def shape_defects(methodology: str, pipeline: dict[str, Any]) -> list[str]:
    """Does this pipeline match the methodology it claims?"""
    if methodology not in REQUIRED:
        return [f"{methodology!r} is not a methodology ({', '.join(sorted(REQUIRED))})"]
    on = {key for key, value in pipeline.items() if value}
    problems = []
    missing = REQUIRED[methodology] - on
    if missing:
        problems.append(f"{methodology} needs {', '.join(sorted(missing))}")
    present = FORBIDDEN.get(methodology, set()) & on
    if present:
        problems.append(f"{methodology} must not have {', '.join(sorted(present))}")
    return problems


def defects(answer: dict[str, Any]) -> list[str]:
    """Everything wrong with this project profile. Empty means it is usable."""
    problems: list[str] = []
    try:
        known = catalog.solution_ids()
    except catalog.Unavailable as error:
        # Not "skip the citation check" — report it. Silently passing every id is how `S-15` got
        # through a validator that returned no defects.
        return [f"cannot check solution ids: {error}"]

    for index, item in enumerate(answer.get("suggested") or []):
        where = f"suggested[{index}]"
        solution = item.get("solution")
        if solution not in known:
            problems.append(f"{where}: cites {solution!r}, which is not in the catalogue")

        methodology = item.get("methodology")
        if methodology not in REQUIRED:
            problems.append(f"{where}: {methodology!r} is not a methodology "
                            f"({', '.join(sorted(REQUIRED))})")
            continue

        on = {key for key, value in (item.get("pipeline_shape") or {}).items() if value}
        missing = REQUIRED[methodology] - on
        if missing:
            problems.append(f"{where}: {methodology} needs {', '.join(sorted(missing))}")
        present = FORBIDDEN.get(methodology, set()) & on
        if present:
            problems.append(f"{where}: {methodology} must not have {', '.join(sorted(present))}")

        if not item.get("why_this_data_supports_it"):
            problems.append(f"{where}: no grounding in the profile's columns")

    for index, item in enumerate(answer.get("declined") or []):
        if item.get("answer") not in {"capture it first", "cannot be done"}:
            problems.append(f"declined[{index}]: {item.get('answer')!r} is neither "
                            "'capture it first' nor 'cannot be done'")
    return problems
