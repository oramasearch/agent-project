# SPDX-License-Identifier: MIT

"""The version in the code has release notes, and the releases index lists them.

The version is read from `pyproject.toml` (and from the installed package, which must agree), never
repeated here, so moving the version without writing its notes fails this test.
"""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASES = ROOT / "docs" / "releases"


def project_version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def test_the_installed_package_reports_the_project_version():
    try:
        installed = version("agent-project")
    except PackageNotFoundError:
        return
    assert installed == project_version()


def test_release_notes_exist_for_the_project_version():
    current = project_version()
    notes = RELEASES / f"v{current}.md"
    assert notes.is_file(), f"no release notes at {notes.relative_to(ROOT)}"
    heading = next(line for line in notes.read_text().splitlines() if line.startswith("# "))
    assert f"v{current}" in heading.split()
    index = (RELEASES / "README.md").read_text()
    assert f"[v{current}](v{current}.md)" in index
