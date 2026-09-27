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

# Which Zetta capability a tool reads from. The build tools do not read — they record an intent
# that lands in the pipeline — so only the reads are mapped. These are the node's real `/v1` reads
# (GET /v1/<capability>): the Pipelines on a node live on its projects, and a node has no explainer
# catalogue, so `list_explainers` observes the models whose drivers can be explained.
READ_CAPABILITY = {
    "list_datasets": "list_datasets",
    "get_dataset": "get_dataset",
    "list_pipelines": "list_projects",
    "list_explainers": "list_models",
}


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

    capability = READ_CAPABILITY.get(call)
    if capability is None:
        # A build step. It records what the agent decided, and the pipeline is where that lands.
        return f"{call} recorded with {args or 'no arguments'}."

    try:
        value = node.read(capability, **args)
    except httpx.HTTPStatusError as error:
        return f"{call} FAILED: {error.response.status_code} {error.response.text[:200]}"
    except httpx.HTTPError as error:
        return f"{call} FAILED: {type(error).__name__}: {error}"
    return f"{call} returned: {value}"
