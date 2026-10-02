"""What a call actually returned.

The corpus generator learned this the expensive way (#989): observations written by the generating
model were coherent by construction, because one author wrote both the observation and the call
that responded to it. That is exactly the property a real node does not have.

The same applies at runtime, more so — an agent that builds a pipeline from observations it
imagined has built it from nothing. So a call goes to the node and the reply is what came back,
including when what came back is a failure.

`--offline` exists for prompt work and says so in every observation. It must never be mistaken for
a run against real data, so it never invents a plausible result: it reports that nothing ran.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

# Which Zetta capability a tool reaches. Reads use GET; compile_intent is the one engine action and
# uses POST. Other build tools record an intent that lands in the returned design. Pipelines on a
# node live on its projects, and a node has no explainer catalogue, so `list_explainers` observes
# the models whose drivers can be explained.
READ_CAPABILITY = {
    "list_solutions": "list_solutions",
    "get_solution": "get_solution",
    "get_policy": "get_policy",
    "get_experiment": "get_experiment",
    "list_ledger": "list_ledger",
    "get_monitor": "get_monitor",
    "get_roi_readout": "get_roi_readout",
    "list_methodologies": "list_methodologies",
    "list_datasets": "list_datasets",
    "get_dataset": "get_dataset",
    "analyze_dataset": "analyze_dataset",
    "plan_features": "plan_features",
    "list_pipelines": "list_projects",
    "list_explainers": "list_models",
}
ACT_CAPABILITY = {
    "create_solution": "create_solution",
    "run_solution": "run_solution",
    "decide": "decide",
    "approve_decision": "approve_decision",
    "compile_intent": "compile_intent",
}


# The arguments Zetta's compile_intent accepts (its input schema refuses any other field). The
# model may still name the person's goal as `intent`; the compiler reads the project's captured
# intent instead, so only these are sent (agent-project#20).
COMPILE_INTENT_ARGS = frozenset({"project", "input_id", "name", "methodology", "solution", "hints"})


class Node:
    """Read/act against a live Zetta node. Raises rather than returning a plausible empty result."""

    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        # The CLI reads the environment; service mode passes the turn's workspace, which in lake is
        # the customer's Zetta through lake (/workspaces/<customer>/zetta) and a lake key.
        self.base = (base_url or os.environ["ZETTA_BASE_URL"]).rstrip("/")
        self.client = httpx.Client(
            base_url=self.base,
            headers={"Authorization": f"Bearer {token or os.environ['ZETTA_TOKEN']}"},
            timeout=180.0)

    def read(self, capability: str, **params: Any) -> Any:
        # get_dataset names its table `dataset` and defaults the database the way lake does.
        if capability == "get_dataset" and "dataset" in params:
            name = str(params["dataset"])
            if "." in name and "database" not in params:
                database, table = name.split(".", 1)
                params = {**params, "database": database, "dataset": table}
            params.setdefault("database", "datasets")
        response = self.client.get(f"/v1/{capability}", params=params)
        response.raise_for_status()
        return response.json().get("value", response.json())

    def act(self, capability: str, **params: Any) -> Any:
        """Run one Zetta action through the workspace-compatible endpoint."""
        response = self.client.post(f"/v1/{capability}", json=params)
        response.raise_for_status()
        return response.json().get("value", response.json())

    def close(self) -> None:
        self.client.close()


def describe(node: Node | None, call: str, args: dict[str, Any]) -> str:
    """Run the call and say what came back, in the model's own vocabulary.

    A failure is returned as an observation rather than raised. The agent is supposed to adapt to
    what the data actually is, and a tool that errors is information about the data — swallowing it
    into an exception would end the trajectory instead of informing it.
    """
    if node is None:
        return (f"OFFLINE: {call} did not run. No observation is available — do not state anything "
                "about the data on the strength of this call.")

    read_capability = READ_CAPABILITY.get(call)
    act_capability = ACT_CAPABILITY.get(call)
    if read_capability is None and act_capability is None:
        # A local design step. It records what the agent decided in the returned pipeline.
        return f"{call} recorded with {args or 'no arguments'}."

    try:
        if read_capability is not None:
            value = node.read(read_capability, **args)
        else:
            if act_capability == "compile_intent":
                args = {key: value for key, value in args.items() if key in COMPILE_INTENT_ARGS}
            value = node.act(act_capability, **args)
    except httpx.HTTPStatusError as error:
        return f"{call} FAILED: {error.response.status_code} {error.response.text[:200]}"
    except httpx.HTTPError as error:
        return f"{call} FAILED: {type(error).__name__}: {error}"
    return f"{call} returned: {value}"
