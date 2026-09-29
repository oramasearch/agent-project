"""Read-only solution turns propose Lake's executable foundation shape."""

from __future__ import annotations

from pathlib import Path

from agent_project import service
from agent_project.retrieval import Corpus


DELIVERY = {
    "kind": "webhook",
    "endpoint": "https://crm.example.test/tasks",
    "owner": "retention-team",
}
ROI = {
    "value_per_outcome": 125,
    "investment": 4000,
    "currency": "USD",
}


def solution(identifier: str) -> dict:
    if identifier == "S-1":
        return {
            "id": "S-1",
            "composition": [
                {"methodology": "M-1", "role": "primary", "minimum_level": 1},
                {"methodology": "M-2", "role": "foundation", "minimum_level": 1},
                {"methodology": "M-3", "role": "foundation", "minimum_level": 1},
            ],
            "policy_template": {
                "eligibility_rules": [
                    {"name": "in_collections"},
                    {"name": "open_complaint"},
                    {"name": "vulnerable"},
                ],
                "exclusion_rules": [],
            },
        }
    return {
        "id": "S-5",
        "composition": [
            {"methodology": "M-4", "role": "primary", "minimum_level": 1},
            {"methodology": "M-2", "role": "foundation", "minimum_level": 1},
            {"methodology": "M-3", "role": "foundation", "minimum_level": 1},
        ],
        "policy_template": {"eligibility_rules": [], "exclusion_rules": []},
    }


def column(name: str, roles: list[str] | None = None, **values) -> dict:
    return {"name": name, "role_candidates": roles or [], **values}


class FoundationNode:
    def __init__(self, identifier: str, include_predicates: bool = True) -> None:
        self.identifier = identifier
        self.include_predicates = include_predicates
        self.calls: list[tuple[str, str, dict]] = []

    def read(self, capability: str, **params):
        self.calls.append(("read", capability, params))
        if capability == "get_solution":
            return {"solution": solution(self.identifier)}
        if capability == "get_dataset":
            columns = [item["name"] for item in self.analysis()["columns"]]
            return {"database": "datasets", "name": "observations", "columns": columns}
        if capability == "analyze_dataset":
            return self.analysis()
        if capability == "plan_features":
            features = ["usage"] if self.identifier == "S-1" else ["promotion"]
            return {
                "status": "ready",
                "methodology": params["methodology"],
                "entries": [{"source_columns": features, "features": features}],
                "excluded": [],
                "transform_required": None,
            }
        raise AssertionError(capability)

    def analysis(self) -> dict:
        if self.identifier == "S-5":
            return {
                "grain": "time_series",
                "candidate_keys": [],
                "columns": [
                    column("series_id", ["entity_key", "group"]),
                    column("period", ["time"], quantiles={"max": 24}),
                    column("demand", ["label"]),
                    column("promotion"),
                    column("product_name"),
                ],
            }
        result = {
            "grain": "snapshot",
            "candidate_keys": [],
            "columns": [
                column("customer_id", ["entity_key"]),
                column(
                    "observed_at",
                    ["time"],
                    top_values=[
                        {"value": "2026-09-30T00:00:00Z", "count": 2},
                        {"value": "2026-08-31T00:00:00Z", "count": 40},
                        {"value": "2026-07-31T00:00:00Z", "count": 40},
                    ],
                ),
                column("churned", ["label"]),
                column("usage"),
                column("account_name"),
            ],
        }
        if self.include_predicates:
            result["columns"].extend([
                column("in_collections"),
                column("open_complaint"),
                column("vulnerable"),
            ])
        return result

    def act(self, capability: str, **params):
        raise AssertionError(f"a foundation turn cannot act: {capability} {params}")

    def close(self) -> None:
        pass


class Issue18Node(FoundationNode):
    def __init__(self) -> None:
        super().__init__("S-1")

    def read(self, capability: str, **params):
        if capability != "plan_features":
            return super().read(capability, **params)
        self.calls.append(("read", capability, params))
        features = [
            "in_collections",
            "open_complaint",
            "vulnerable",
            "tenure_months",
            "usage",
        ]
        return {
            "status": "ready",
            "methodology": params["methodology"],
            "entries": [{"source_columns": features, "features": features}],
            "excluded": [],
            "transform_required": None,
        }

    def analysis(self) -> dict:
        return {
            "grain": "snapshot",
            "candidate_keys": [
                {"columns": ["monthly_value"], "uniqueness": 1.0},
                {"columns": ["customer_id", "observed_at"], "uniqueness": 1.0},
            ],
            "columns": [
                column(
                    "customer_id",
                    ["entity_key"],
                    semantic_type="identifier",
                    distinct_count=40,
                    distinct_ratio=0.5,
                ),
                column(
                    "observed_at",
                    ["time"],
                    semantic_type="datetime",
                    distinct_count=2,
                    distinct_ratio=0.025,
                    top_values=[
                        {"value": "2026-09-30T00:00:00Z", "count": 40},
                        {"value": "2026-08-31T00:00:00Z", "count": 40},
                    ],
                ),
                column("usage", semantic_type="numeric", quantiles={"max": 98.0}),
                column("tenure_months", semantic_type="numeric", quantiles={"max": 72.0}),
                column("complaints", semantic_type="numeric", quantiles={"max": 8.0}),
                column(
                    "monthly_value",
                    semantic_type="identifier",
                    distinct_count=80,
                    distinct_ratio=1.0,
                    quantiles={"min": 19.99, "max": 249.95},
                ),
                column("in_collections", semantic_type="boolean"),
                column("open_complaint", semantic_type="boolean"),
                column("vulnerable", semantic_type="boolean"),
                column("churned", ["label"], semantic_type="boolean"),
            ],
        }


def settings(tmp_path: Path) -> service.Settings:
    value = service.Settings()
    value.trace_dir = tmp_path / "traces"
    return value


def turn(solution_id: str, project: str = "retention") -> dict:
    return {
        "kind": f"solution:{project}",
        "request": "Propose the executable solution foundation.",
        "context": {
            "project": project,
            "solution": solution_id,
            "level": 1,
            "dataset": "datasets.observations",
            "delivery": DELIVERY,
            "roi_model": ROI,
        },
    }


def run(tmp_path: Path, node: FoundationNode, value: dict) -> dict:
    return service.run_turn(
        value,
        settings(tmp_path),
        Corpus(tmp_path / "missing.jsonl"),
        node=node,
    )


def test_s1_foundation_maps_observed_roles_predicates_and_complete_period(tmp_path):
    node = FoundationNode("S-1")

    answer = run(tmp_path, node, turn("S-1"))

    assert answer["ok"] is True
    result = answer["result"]
    foundation = result["foundation"]
    assert result["read_only"] is True
    assert foundation["snapshot"] == {
        "database": "datasets",
        "source_table": "observations",
        "target_table": "retention_snapshot",
        "entity_column": "customer_id",
        "time_column": "observed_at",
        "as_of": "2026-08-31T00:00:00Z",
    }
    assert foundation["dataset"] == {
        "entity_column": "customer_id",
        "label_column": "churned",
        "feature_columns": ["usage"],
    }
    assert set(foundation["sources"]) == {"source", "table", "M-1", "M-2", "M-3"}
    assert foundation["sources"]["M-2"] == {
        "source": "datasets",
        "table": "observations",
        "record_id": "customer_id",
        "name": "account_name",
    }
    assert foundation["predicates"] == {
        "in_collections": "in_collections",
        "open_complaint": "open_complaint",
        "vulnerable": "vulnerable",
    }
    assert foundation["missing_predicates"] == []
    assert foundation["delivery"] == DELIVERY
    assert foundation["roi_model"] == ROI
    assert all(call[0] == "read" for call in node.calls)
    assert [call[1] for call in node.calls] == [
        "get_solution",
        "analyze_dataset",
        "get_dataset",
        "plan_features",
        "plan_features",
        "plan_features",
    ]


def test_s1_foundation_does_not_invent_absent_exclusion_flags(tmp_path):
    answer = run(tmp_path, FoundationNode("S-1", include_predicates=False), turn("S-1"))

    foundation = answer["result"]["foundation"]
    assert foundation["predicates"] == {}
    assert foundation["missing_predicates"] == [
        "in_collections",
        "open_complaint",
        "vulnerable",
    ]


def test_s1_foundation_prefers_customer_id_and_keeps_predicates_out_of_features(tmp_path):
    answer = run(tmp_path, Issue18Node(), turn("S-1"))

    result = answer["result"]
    foundation = result["foundation"]
    expected_features = ["usage", "tenure_months", "complaints", "monthly_value"]
    assert foundation["snapshot"]["entity_column"] == "customer_id"
    assert foundation["dataset"] == {
        "entity_column": "customer_id",
        "label_column": "churned",
        "feature_columns": expected_features,
    }
    assert foundation["sources"]["M-1"]["entity_column"] == "customer_id"
    assert foundation["sources"]["M-1"]["feature_columns"] == expected_features
    assert foundation["sources"]["M-2"] == {
        "source": "datasets",
        "table": "observations",
        "record_id": "customer_id",
    }
    assert "No observed column mapping supports M-2.name." in result["reply"]


def test_s5_foundation_names_the_forecast_output_as_its_decision_source(tmp_path):
    node = FoundationNode("S-5")

    answer = run(tmp_path, node, turn("S-5", project="demand-plan"))

    foundation = answer["result"]["foundation"]
    assert foundation["decision_source"] == {
        "methodology": "M-4",
        "entity_column": "series",
        "score_column": "point_forecast",
        "output_name": "forecast",
    }
    assert foundation["sources"]["M-4"] == {
        "time_column": "period",
        "target_column": "demand",
        "series_column": "series_id",
    }
    assert foundation["dataset"] == {
        "entity_column": "series_id",
        "label_column": "demand",
        "feature_columns": ["promotion"],
    }
    assert set(foundation["sources"]) == {"source", "table", "M-4", "M-2", "M-3"}
    assert all(call[0] == "read" for call in node.calls)
