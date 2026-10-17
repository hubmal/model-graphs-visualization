"""Common interface for transformation passes."""

from abc import ABC, abstractmethod
from typing import ClassVar, Self

from graph_viewer.graph import Graph


class PassConfigError(ValueError):
    """Raised for an unknown pass name or invalid pass options."""


class TransformationPass(ABC):
    """A step that changes how a Graph is visualized.

    Passes modify the graph in place and return it (no full copy per pass). They only
    see the intermediate Graph, so the original model is never touched.
    """

    # Name used on the command line and in the registry, e.g. "hide-constants".
    name: ClassVar[str]
    # One-line description shown by `--list-passes`.
    description: ClassVar[str] = ""

    @abstractmethod
    def apply(self, graph: Graph) -> Graph:
        """Transform `graph` and return it."""

    @classmethod
    def from_options(cls, options: dict[str, str]) -> Self:
        """Create the pass from string options (e.g. from the CLI).

        The default accepts no options. Passes with options override this.
        """
        if options:
            raise PassConfigError(f"pass '{cls.name}' takes no options, got: {', '.join(options)}")
        return cls()
