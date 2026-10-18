"""Abstract Renderer interface shared by all output formats."""

from abc import ABC, abstractmethod
from pathlib import Path

from graph_viewer.graph import Graph


class RenderError(Exception):
    """Raised when the output can't be produced (e.g. a missing external tool)."""


class Renderer(ABC):
    """Turns a Graph into an output file."""

    @abstractmethod
    def render(self, graph: Graph, output_path: str | Path) -> Path:
        """Write the visualization of `graph` to `output_path` and return the path."""
