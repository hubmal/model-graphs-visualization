"""Public Python API: load a model, apply passes, render an SVG."""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from graph_viewer.graph import Graph
from graph_viewer.loaders import get_loader
from graph_viewer.passes import Pipeline, TransformationPass, create_pass
from graph_viewer.renderers import GraphvizRenderer, Renderer


@dataclass
class VisualizationResult:
    """What visualize() produced, with graph sizes and the time spent in each stage."""

    output_path: Path
    nodes_before: int
    edges_before: int
    nodes_after: int
    edges_after: int
    load_seconds: float
    passes_seconds: float
    render_seconds: float
    pass_timings: list[tuple[str, float]] = field(default_factory=list)


def load_model(path: str | Path) -> Graph:
    """Load a model file into a Graph, choosing the loader by file extension."""
    return get_loader(path).load(path)


def visualize(
    model_path: str | Path,
    output_path: str | Path,
    passes: Sequence[TransformationPass | str] = (),
    renderer: Renderer | None = None,
) -> VisualizationResult:
    """Load `model_path`, apply `passes` in order and write the SVG to `output_path`.

    `passes` may contain pass objects or registered pass names (without options).
    """
    pipeline = Pipeline([create_pass(p) if isinstance(p, str) else p for p in passes])
    renderer = renderer or GraphvizRenderer()

    start = time.perf_counter()
    graph = load_model(model_path)
    load_seconds = time.perf_counter() - start
    nodes_before, edges_before = graph.num_nodes(), graph.num_edges()

    start = time.perf_counter()
    graph = pipeline.run(graph)
    passes_seconds = time.perf_counter() - start

    start = time.perf_counter()
    path = renderer.render(graph, output_path)
    render_seconds = time.perf_counter() - start

    return VisualizationResult(
        output_path=path,
        nodes_before=nodes_before,
        edges_before=edges_before,
        nodes_after=graph.num_nodes(),
        edges_after=graph.num_edges(),
        load_seconds=load_seconds,
        passes_seconds=passes_seconds,
        render_seconds=render_seconds,
        pass_timings=pipeline.timings,
    )
