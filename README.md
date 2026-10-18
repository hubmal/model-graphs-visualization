# Model Graph Viewer

A small Python library and CLI that loads a neural network model (OpenVINO IR), converts it
into a framework-independent graph, applies configurable transformation passes, and renders
the result as an SVG with Graphviz.

```
model.xml ──► OpenVINOLoader ──► Graph ──► passes (in order) ──► GraphvizRenderer ──► graph.svg
```

- Ports are preserved: every edge knows `out <i> → in <j>`, plus dtype and (dynamic) shape.
- Built-in passes: `highlight`, `hide-constants`, `fold-converts` (custom, see below).
- New passes and model formats are added without touching existing components.

## Getting started

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and the Graphviz `dot` binary
(`sudo apt install graphviz`, `brew install graphviz`, ...).

```bash
git clone <this repo> && cd model-graphs-visualization
uv sync                      # library + CLI + test dependencies
uv run pytest                # run the tests
```

Get example models (needs the `export` extra: optimum-intel and CPU-only PyTorch):

```bash
uv sync --extra export
uv run python scripts/export_models.py              # all three models into ml_models/
uv run python scripts/export_models.py tinyllama    # or just one
```

| Model | Source | Export |
|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | Hugging Face | `optimum-cli export openvino --weight-format fp16` (as in the assignment) |
| DistilBERT base uncased | Hugging Face | `optimum-cli export openvino --task feature-extraction` |
| ResNet-18 | torchvision (random weights; structure is identical) | `openvino.convert_model` |

Render your first graph:

```bash
uv run python model_graph_viewer.py \
    --model ml_models/resnet18_ov/openvino_model.xml \
    --output resnet18.svg \
    --passes fold-converts,hide-constants
```

Open `resnet18.svg` in a browser. Hover a node to see its full name.

## Usage

### CLI

```
python model_graph_viewer.py --model MODEL.xml --output OUT.svg
                             [--passes P1,P2,...] [-O PASS.KEY=VALUE ...]
                             [--no-attributes] [--list-passes]
```

| Option | Meaning |
|---|---|
| `--model` | Input model (OpenVINO IR `.xml`; the `.bin` must be next to it). |
| `--output` | Output SVG path (folders are created). |
| `--passes` | Comma-separated pass names, applied **in this order**. Omit for no passes. |
| `-O`, `--pass-option` | Option for a selected pass, `PASS.KEY=VALUE`; repeatable. |
| `--no-attributes` | Show only op type, name and id in nodes. |
| `--list-passes` | Print the available passes and exit. |

Examples:

```bash
# No transformations
python model_graph_viewer.py --model model.xml --output graph.svg

# Simplify: fold compressed weights, then hide constants
python model_graph_viewer.py --model model.xml --output simple.svg \
    --passes fold-converts,hide-constants

# Highlight nodes by name (comma-separated), optionally in another color
python model_graph_viewer.py --model model.xml --output hl.svg \
    --passes hide-constants,highlight \
    -O "highlight.names=__module.layer1.0/aten::add_/Add,__module.layer1.1/aten::add_/Add" \
    -O highlight.color=blue
```

The CLI prints graph sizes and the time per stage:

```
wrote simple.svg
graph:  98 nodes, 105 edges -> 52 nodes, 59 edges
load:   0.010 s
pass:   0.000 s  fold-converts
pass:   0.000 s  hide-constants
render: 0.028 s
```

Invalid input gives a one-line message and exit code 1 (exit code 2 for wrong CLI usage):

```
error: nope.xml: file not found
error: unknown pass 'hide-constnts' (available: fold-converts, hide-constants, highlight)
error: a.onnx: unsupported model format '.onnx' (supported: .xml)
```

### Passes

| Name | What it does | Options |
|---|---|---|
| `highlight` | Red (or `color`) thick border on nodes whose name is in `names`. Warns about names not found. | `names` (required, comma-separated), `color` |
| `hide-constants` | Removes `Constant` nodes and adds `[port]: dtype[shape]` to each consumer, e.g. `[1]: f32[1024, 768]`. A Constant with several consumers adds `(shared: <constant id>)`. Tensor values are never read. | — |
| `fold-converts` | Removes `Convert` nodes fed only by a `Constant` (compressed weights) and connects the Constant directly, keeping the stored dtype. Run it **before** `hide-constants`. | — |

### Python API

```python
from graph_viewer.api import load_model, visualize
from graph_viewer.passes import FoldConstantConvertsPass, HideConstantsPass, HighlightPass, Pipeline
from graph_viewer.renderers import GraphvizRenderer

# One call: load -> passes -> SVG. Passes may be objects or registered names.
result = visualize(
    "ml_models/tinyllama_ov/openvino_model.xml",
    "tinyllama.svg",
    passes=["fold-converts", "hide-constants",
            HighlightPass(["__module.model.layers.0.self_attn.q_proj/ov_ext::linear/MatMul"])],
)
print(result.nodes_before, "->", result.nodes_after, "nodes")
print(result.load_seconds, result.passes_seconds, result.render_seconds)

# Step by step, with full control over each component
graph = load_model("ml_models/resnet18_ov/openvino_model.xml")
graph = Pipeline([FoldConstantConvertsPass(), HideConstantsPass()]).run(graph)
for node in graph.nodes():
    print(node.node_id, node.op_type, node.name, node.attributes)
GraphvizRenderer(show_attributes=False).render(graph, "resnet18.svg")
dot_source = GraphvizRenderer().to_dot(graph)   # DOT text, no Graphviz needed
```

Passes can also be built from strings, the way the CLI does it:
`create_pass("highlight", {"names": "a,b", "color": "blue"})`.

## Examples

Generated by `uv run python scripts/benchmark.py` into [`examples/`](examples/):

| Model | Original | Simplified | Other |
|---|---|---|---|
| TinyLlama | [original.svg](examples/tinyllama/original.svg) | [simplified.svg](examples/tinyllama/simplified.svg) | [hide-constants only](examples/tinyllama/hide-constants.svg), [layer 0 q/k/v highlighted](examples/tinyllama/highlight-qkv.svg) |
| DistilBERT | [original.svg](examples/distilbert/original.svg) | [simplified.svg](examples/distilbert/simplified.svg) | |
| ResNet-18 | [original.svg](examples/resnet18/original.svg) | [simplified.svg](examples/resnet18/simplified.svg) | [hide-constants only](examples/resnet18/hide-constants.svg), [residual Adds highlighted](examples/resnet18/highlight-residual-adds.svg) |

"Simplified" = `fold-converts,hide-constants`.

### Custom pass: why `fold-converts`

In Netron, TinyLlama (exported with `--weight-format fp16`) shows every weight as a
**Constant → Convert** pair in front of its MatMul: 1257 Constants and 388 Converts out of
3168 operations. The Converts only decompress f16 weights to f32 at runtime and carry no
information about the computation. Worse, they block constant hiding: the MatMul's weight
input comes from a Convert, so `hide-constants` alone leaves all weights visible as
Constant → Convert chains.

`fold-converts` removes such a Convert and connects the Constant directly to its consumers,
so `hide-constants` can then fold the weight into the MatMul as `[1]: f16[2048, 2048]`:

| TinyLlama | Nodes | Edges |
|---|---|---|
| original | 3168 | 3741 |
| `hide-constants` | 1911 | 2429 |
| `fold-converts,hide-constants` | 1528 | 2025 |

The result shows the main computation (MatMul, Add, Multiply, attention ops) with the
weight shapes written on the operations. Compare
[hide-constants.svg](examples/tinyllama/hide-constants.svg) with
[simplified.svg](examples/tinyllama/simplified.svg). ResNet-18 benefits the same way
(98 → 74 → 52 nodes).

## Architecture

```
   loaders/            graph.py            renderers/
 (OpenVINO, ...)  ───►  Node, Edge,  ◄───  (Graphviz SVG)
                        Graph
                          ▲
                       passes/
            (highlight, hide-constants, fold-converts)
```

```
src/graph_viewer/
├── graph.py               # Node, Edge, Graph (no framework imports)
├── loaders/
│   ├── base.py            # ModelLoader (abstract), ModelLoadError
│   ├── openvino_loader.py # OpenVINO IR -> Graph
│   └── __init__.py        # LOADERS: file extension -> loader
├── passes/
│   ├── base.py            # TransformationPass (abstract), PassConfigError
│   ├── registry.py        # @register_pass, create_pass()
│   ├── pipeline.py        # runs passes in order, records timings
│   └── highlight.py, hide_constants.py, fold_converts.py
├── renderers/
│   ├── base.py            # Renderer (abstract), RenderError
│   └── graphviz_renderer.py
├── api.py                 # load_model(), visualize()
└── cli.py                 # argument parsing on top of api.py
model_graph_viewer.py      # CLI entry point
```

Key decisions:

- **Ports & Adapters:** `graph.py` is the core and imports nothing else; loaders (input
  adapters) and renderers (output adapters) depend on it, never the other way round. Tests
  check that `graph_viewer.graph` and `graph_viewer.passes` don't import OpenVINO or Graphviz.
- **Graph = dicts + per-node edge indexes** (`node -> {input port: edge}`,
  `node -> {edge ids}`): lookups and modifications cost O(degree), never a scan of all
  nodes/edges. Node ids are generated by the graph, separate from names (names may repeat),
  and never reused. Multiple edges between the same nodes are distinct (different ports).
- **Shapes:** `(None, 2048)` = `[?, 2048]`, `None` = unknown rank / no metadata, `()` = scalar.
- **Passes modify the graph in place** (no copy per pass); the original model is never touched,
  because passes only see the intermediate graph.
- **Visualization metadata** is a free-form `node.viz` dict (`border_color`, `fill_color`);
  the renderer applies the keys it knows.

## Extensibility

### Add a transformation pass

Create `src/graph_viewer/passes/my_pass.py`:

```python
from graph_viewer.graph import Graph
from graph_viewer.passes.base import TransformationPass
from graph_viewer.passes.registry import register_pass


@register_pass
class DropResultsPass(TransformationPass):
    name = "drop-results"
    description = "Remove Result nodes."

    def apply(self, graph: Graph) -> Graph:
        for node in graph.nodes():          # snapshot, safe to modify while iterating
            if node.op_type == "Result":
                graph.remove_node(node.node_id)
        return graph
```

Import it in `passes/__init__.py` (one line). It is now available as
`--passes drop-results` and `create_pass("drop-results")`. Passes with options override
`from_options(cls, options: dict[str, str])` (see `HighlightPass`). The parser and renderer
don't change.

### Add a model format (e.g. ONNX)

1. Write `loaders/onnx_loader.py` with `class ONNXLoader(ModelLoader)` whose `load(path)`
   builds a `Graph` with `add_node(name, op_type, attributes)` and
   `add_edge(src, src_port, dst, dst_port, dtype, shape)`, raising `ModelLoadError` for bad files.
2. Register it in `loaders/__init__.py`: `LOADERS[".onnx"] = ONNXLoader`.

Passes, renderer, API and CLI work unchanged.

## Testing

```bash
uv run pytest              # all tests
uv run pytest -k loader    # a subset
```

| File | Covers |
|---|---|
| `tests/test_graph.py` | nodes/edges, ports, multi-edges, duplicate names, dynamic shapes, removal keeps indexes consistent, 20k-node scale test, no framework imports |
| `tests/test_openvino_loader.py` | small OpenVINO models built in code: types, names, ports, dtypes, (dynamic) shapes, attributes without tensor values, errors (missing/invalid file, missing `.bin`); TinyLlama if exported |
| `tests/test_passes.py` | highlight, hide-constants (single/shared consumers, port order, metadata edge cases), fold-converts, pipeline order, registry and options |
| `tests/test_renderer.py` | DOT content (labels, ports, escaping, highlight, unknown ops) and valid SVG output |
| `tests/test_api_cli.py` | end-to-end OpenVINO model → passes → SVG through the API and CLI; error messages |

Transformations are tested on small synthetic graphs, independent of the loader and
renderer. `tests/graph_checks.py::assert_graph_consistent` verifies after modifications that
the per-node indexes match the edge set. SVG tests are skipped if `dot` is not installed;
the TinyLlama test is skipped if the model is not exported.

## Performance

Measured with `uv run python scripts/benchmark.py` (one run each; this also regenerates
`examples/`).

**Environment:** AMD EPYC (Genoa) VM, 128 logical CPUs, Python 3.11.13, OpenVINO 2026.4.1, Graphviz 2.43.0, Linux 5.15.

| Model | Passes | Nodes | Edges | Load (s) | Passes (s) | Render (s) | SVG size |
|---|---|---|---|---|---|---|---|
| tinyllama | none | 3168 → 3168 | 3741 → 3741 | 0.156 | 0.0000 | 171.97 | 5.6 MB |
| tinyllama | hide-constants | 3168 → 1911 | 3741 → 2429 | 0.126 | 0.0050 | 183.97 | 3.6 MB |
| tinyllama | fold-converts, hide-constants | 3168 → 1528 | 3741 → 2025 | 0.131 | 0.0070 | 69.62 | 3.0 MB |
| tinyllama | fold-converts, hide-constants, highlight | 3168 → 1528 | 3741 → 2025 | 0.130 | 0.0086 | 71.19 | 3.0 MB |
| distilbert | none | 438 → 438 | 483 → 483 | 0.017 | 0.0000 | 0.38 | 0.8 MB |
| distilbert | fold-converts, hide-constants | 438 → 235 | 483 → 274 | 0.017 | 0.0007 | 0.15 | 0.4 MB |
| resnet18 | none | 98 → 98 | 105 → 105 | 0.006 | 0.0000 | 0.03 | 0.2 MB |
| resnet18 | hide-constants | 98 → 74 | 105 → 81 | 0.014 | 0.0001 | 0.03 | 0.1 MB |
| resnet18 | fold-converts, hide-constants | 98 → 52 | 105 → 59 | 0.006 | 0.0002 | 0.03 | 0.1 MB |
| resnet18 | fold-converts, hide-constants, highlight | 98 → 52 | 105 → 59 | 0.005 | 0.0002 | 0.03 | 0.1 MB |

Times are wall-clock seconds per stage: *Load* = reading the IR and building the graph,
*Passes* = all passes together, *Render* = Graphviz layout + SVG writing. TinyLlama graphs
have more than 1000 nodes, so they are rendered with the fast layout settings (see below).

**Observations**

- Loading and all passes take milliseconds even for TinyLlama: graph operations are
  O(degree) thanks to the per-node indexes, and OpenVINO memory-maps the 2.2 GB `.bin`
  without reading tensor values.
- **Rendering with Graphviz `dot` dominates** the total time and grows much faster than
  linearly with graph size. Measured on TinyLlama, the bottleneck is dot's *network simplex*
  (layer assignment and x-coordinates): with default settings neither the raw nor the
  simplified graph finished in 10 minutes. Limiting its iterations
  (`nslimit=2 nslimit1=2 mclimit=0.3`) makes it finish, and straight edges
  (`splines=line`) avoid a crash of Graphviz 2.43's curved-edge routing on graphs this big.
  The renderer applies these settings automatically above 1000 nodes
  (`GraphvizRenderer(fast_layout_threshold=...)`). Removing edge labels did not help on its
  own:

  | dot variant (TinyLlama, 10 min cap) | raw (3168 nodes) | simplified (1528 nodes) |
  |---|---|---|
  | default settings | > 600 s | > 600 s |
  | no edge labels | > 600 s | > 600 s |
  | `nslimit=2 nslimit1=2 mclimit=0.3` (curved edges) | 174 s | 74 s |
  | the above + `splines=line` (used) | 158 s | 64 s |

  The curved-edge variant finished in the first experiment, but crashed dot 2.43 with SIGBUS
  on both graphs when the benchmark was re-run, so straight edges are used.
- Simplification pays off in rendering: `fold-converts,hide-constants` cuts TinyLlama's render
  time from 172 s to 70 s and the SVG from 5.6 MB to 3.0 MB. `hide-constants` alone renders
  no faster (184 s) despite 40% fewer nodes, so layout time depends on the graph's structure,
  not only on its node count. I did not profile this case further.

## Known limitations and possible improvements

- **Graphviz scalability:** layouts of thousands of nodes take minutes even with the fast
  settings, the fast layout is less tidy, and the SVG is very tall. Possible improvements:
  collapse repeated blocks (e.g. the 22 identical decoder layers of TinyLlama) into single
  nodes, render only a subgraph around selected nodes, or use a faster layout engine
  (e.g. ELK or a WebGL viewer).
- **Sub-graphs** of ops like `Loop`/`TensorIterator`/`If` are shown as single nodes; their
  bodies are not expanded.
- **Only OpenVINO IR** is implemented; other formats need a loader (see Extensibility).
- `fold-converts` labels the folded edge with the stored dtype (e.g. `f16`), not the
  converted one (`f32`).
- Bounded dynamic dimensions (e.g. `1..10`) are shown as `?`.
- Highlighting matches exact names only (no patterns).
