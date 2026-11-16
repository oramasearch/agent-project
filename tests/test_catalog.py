# SPDX-License-Identifier: MIT

"""The catalogue an image carries (ZETTA_CATALOG) is read as built, and an empty one is refused."""

import json

import pytest

from agent_project import catalog


def test_a_built_catalogue_is_read_and_an_empty_one_refused(tmp_path, monkeypatch):
    built = tmp_path / "catalog.json"
    built.write_text(json.dumps({"solutions": [{"id": "S-1", "name": "Churn"}]}))
    monkeypatch.setenv("ZETTA_CATALOG", str(built))
    catalog.load.cache_clear()
    assert catalog.solution_ids() == {"S-1"}
    built.write_text(json.dumps({"solutions": []}))
    catalog.load.cache_clear()
    with pytest.raises(catalog.Unavailable):
        catalog.load()
    catalog.load.cache_clear()
