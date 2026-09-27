"""The project agent — builds and changes Pipelines by calling tools."""

__all__ = ["main", "build", "serve"]

from agent_project.agent import build, main
from agent_project.service import serve
