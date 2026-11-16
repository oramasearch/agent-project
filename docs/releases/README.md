# Releases

One page per tagged release, newest first. A page states what the release includes, what it does
not include, known issues, verification output, and where the image runs. The first release also
lists every turn, tool, route, command and environment variable.

A release has one `vX.Y.Z` name shared by the Git tag, the GitHub release and the Docker Hub
image tag `oramasearch/agent-project:vX.Y.Z`. The image tag names the digest already built from the
tag and deployed; it is never produced by a second build.

| Release | Date | Summary |
|---|---|---|
| [v0.2.0](v0.2.0.md) | 2026-10-25 | A minor release: the service token always required and compared in constant time, a turn's credential only to `LAKE_URL`, and a Lake turn that designs a pipeline or returns a `solution_proposal` without a write (prompt v7) |
| [v0.1.0](v0.1.0.md) | 2026-10-17 | The first release: solution operation, methodology design, edit proposals and solution foundations as Lake library-agent turns over the agent bus, the model through Lake's gateway, App and Experiments handoffs, and the image |

`pyproject.toml`'s `[project].version` is the one place the version is set; `/healthz` reports it,
and `tests/test_release.py` fails when that version has no page here.
