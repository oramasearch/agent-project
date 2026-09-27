"""The solutions catalogue, as context rather than as a tool.

The eval generator taught this the expensive way: when the catalogue was reachable only through a
tool, every run anchored on whatever that tool happened to return first, and three independent runs
produced the same domain. Handed to the model as context, the whole catalogue is in view at once
and nothing is privileged by call order.

It is also the vocabulary. Without it the model invents plausible identifiers — an early run cited
`S-23` against a catalogue whose highest id is S-12. A citation nobody can look up is worse than no
citation, because it reads as grounded.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

BUILDER = "modules/zetta/scripts/build_catalog.py"


class Unavailable(RuntimeError):
    """The catalogue could not be built.

    Raised rather than returned-as-empty. An empty catalogue makes `solution_ids()` empty, which
    makes the citation check vacuously pass — the agent then invents ids and the validator reports
    a clean answer. A missing catalogue must fail loudly or it disables the check that catches the
    failure it causes.
    """


def _builder() -> Path:
    # An explicit ZETTA_BRIDGE is authoritative. Falling back to the parent walk when it points
    # somewhere wrong would build a DIFFERENT catalogue than the one asked for and report success.
    override = os.environ.get("ZETTA_BRIDGE")
    if override:
        builder = Path(override) / BUILDER
        if not builder.exists():
            raise Unavailable(f"ZETTA_BRIDGE={override} has no {BUILDER}")
        return builder
    for root in Path(__file__).resolve().parents:
        if (root / BUILDER).exists():
            return root / BUILDER
    raise Unavailable(
        f"no {BUILDER} above {Path(__file__).parent} — set ZETTA_BRIDGE to the bridge checkout")


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    # An image carries the catalogue its build produced with this same builder (ZETTA_CATALOG),
    # beside a catalog.source.json naming the commits it came from. Still derived, never retyped.
    built = os.environ.get("ZETTA_CATALOG")
    if built:
        try:
            catalogue = json.loads(Path(built).read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise Unavailable(f"ZETTA_CATALOG={built} cannot be read: {error}") from error
        if not catalogue.get("solutions"):
            raise Unavailable(f"ZETTA_CATALOG={built} has no solutions")
        return catalogue
    return _build()


def _build() -> dict[str, Any]:
    """Build the catalogue from the plan markdown — the same builder the engine uses.

    Deriving it beats vendoring a copy: a catalogue snapshot in this repo would drift from the plan
    silently, and the first symptom would be the agent citing a solution that no longer exists.
    """
    # `uv run --script` because the builder carries PEP 723 metadata — it declares jsonschema and
    # will not import under a bare interpreter that lacks it. Running it that way keeps its
    # dependency out of ours.
    builder = _builder()
    command = ([shutil.which("uv"), "run", "--script", str(builder)] if shutil.which("uv")
               else [sys.executable, str(builder)])
    result = subprocess.run([*command, "--out", "/dev/stdout"],
                            capture_output=True, text=True, check=False)
    body = result.stdout[result.stdout.find("{"): result.stdout.rfind("}") + 1]
    try:
        catalogue = json.loads(body)
    except json.JSONDecodeError as error:
        raise Unavailable(f"build_catalog failed: {result.stderr.strip()[:200]}") from error
    if not catalogue.get("solutions"):
        raise Unavailable("build_catalog returned no solutions")
    return catalogue


def solution_ids() -> set[str]:
    return {s["id"] for s in load().get("solutions", [])}


def as_context() -> str:
    """The catalogue as prompt text. Ids and names only — the full text of twelve solutions would
    crowd out the profile, and the id is what has to be right."""
    catalogue = load()
    solutions = catalogue["solutions"]
    lines = [f"- {s['id']}: {s['name']}" + (f" [{s['uc']}]" if s.get("uc") else "")
             for s in solutions]
    methods = catalogue.get("methodologies", {}).get("methods", [])
    return ("\n\n## The solutions catalogue — cite only these ids\n\n"
            + "\n".join(lines)
            + "\n\nIf nothing here fits, say so in `declined` rather than inventing an id.\n"
            + "\n## Methodology families\n\n"
            + "\n".join(f"- {m['id']}: {m['name']}" for m in methods))
