"""Runs a configurable sequence of passes in order."""

import time
from collections.abc import Sequence

from graph_viewer.graph import Graph
from graph_viewer.passes.base import TransformationPass


class Pipeline:
    def __init__(self, passes: Sequence[TransformationPass]) -> None:
        self.passes = list(passes)
        # (pass name, seconds) for each pass of the most recent run().
        self.timings: list[tuple[str, float]] = []

    def run(self, graph: Graph) -> Graph:
        self.timings = []
        for transformation_pass in self.passes:
            start = time.perf_counter()
            graph = transformation_pass.apply(graph)
            self.timings.append((transformation_pass.name, time.perf_counter() - start))
        return graph
