"""Generate the example SVGs in examples/ and print a benchmark table (Markdown).

Needs the exported models (scripts/export_models.py).
Usage: uv run python scripts/benchmark.py
"""

import os
import platform
from pathlib import Path

import graphviz
import openvino

from graph_viewer.api import visualize
from graph_viewer.passes import HighlightPass

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "ml_models"
EXAMPLES = ROOT / "examples"

LLAMA_LAYER0 = "__module.model.layers.0.self_attn.{}_proj/ov_ext::linear/MatMul"

# model directory -> list of (example name, passes)
RUNS = {
    "tinyllama_ov": [
        ("original", []),
        ("hide-constants", ["hide-constants"]),
        ("simplified", ["fold-converts", "hide-constants"]),
        ("highlight-qkv", ["fold-converts", "hide-constants",
                           HighlightPass([LLAMA_LAYER0.format(p) for p in ("q", "k", "v")])]),
    ],
    "distilbert_ov": [
        ("original", []),
        ("simplified", ["fold-converts", "hide-constants"]),
    ],
    "resnet18_ov": [
        ("original", []),
        ("hide-constants", ["hide-constants"]),
        ("simplified", ["fold-converts", "hide-constants"]),
        ("highlight-residual-adds", ["fold-converts", "hide-constants",
                                     HighlightPass([f"__module.layer{i}.{j}/aten::add_/Add"
                                                    for i in range(1, 5) for j in range(2)])]),
    ],
}


def main() -> None:
    print(f"Environment: {platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
          f"Python {platform.python_version()}, OpenVINO {openvino.__version__.split('-')[0]}, "
          f"Graphviz {'.'.join(map(str, graphviz.version()))}\n")
    print("| Model | Passes | Nodes | Edges | Load (s) | Passes (s) | Render (s) | SVG size |")
    print("|---|---|---|---|---|---|---|---|")
    for model_dir, runs in RUNS.items():
        model = MODELS / model_dir / "openvino_model.xml"
        for example_name, passes in runs:
            output = EXAMPLES / model_dir.removesuffix("_ov") / f"{example_name}.svg"
            r = visualize(model, output, passes)
            names = ", ".join(name for name, _ in r.pass_timings) or "none"
            print(f"| {model_dir.removesuffix('_ov')} | {names} "
                  f"| {r.nodes_before} → {r.nodes_after} | {r.edges_before} → {r.edges_after} "
                  f"| {r.load_seconds:.3f} | {r.passes_seconds:.4f} | {r.render_seconds:.2f} "
                  f"| {output.stat().st_size / 1e6:.1f} MB |", flush=True)


if __name__ == "__main__":
    main()
