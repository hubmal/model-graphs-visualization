"""Highlight nodes by name."""

import logging
from collections.abc import Iterable
from typing import Self

from graph_viewer.graph import Graph
from graph_viewer.passes.base import PassConfigError, TransformationPass
from graph_viewer.passes.registry import register_pass

logger = logging.getLogger(__name__)


@register_pass
class HighlightPass(TransformationPass):
    name = "highlight"
    description = "Draw a colored border around nodes with the given names (options: names, color)."

    def __init__(self, names: Iterable[str], color: str = "red") -> None:
        self.names = set(names)
        self.color = color

    @classmethod
    def from_options(cls, options: dict[str, str]) -> Self:
        unknown = set(options) - {"names", "color"}
        if unknown:
            raise PassConfigError(f"pass '{cls.name}': unknown options: {', '.join(sorted(unknown))}")
        names = [n.strip() for n in options.get("names", "").split(",") if n.strip()]
        if not names:
            raise PassConfigError(f"pass '{cls.name}' needs names, e.g. highlight.names=MatMul_1,Add_2")
        return cls(names, options.get("color", "red"))

    def apply(self, graph: Graph) -> Graph:
        found: set[str] = set()
        for node in graph.nodes():
            if node.name in self.names:
                node.viz["border_color"] = self.color
                found.add(node.name)
        if missing := self.names - found:
            logger.warning("highlight: no node named %s", ", ".join(sorted(missing)))
        return graph
