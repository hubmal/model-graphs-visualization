"""Command-line interface built on top of graph_viewer.api."""

import argparse
import logging
import sys

from graph_viewer.api import visualize
from graph_viewer.loaders import ModelLoadError
from graph_viewer.passes import PassConfigError, TransformationPass, available_passes, create_pass
from graph_viewer.renderers import GraphvizRenderer, RenderError

EXAMPLES = """\
examples:
  %(prog)s --model model.xml --output graph.svg
  %(prog)s --model model.xml --output simple.svg --passes fold-converts,hide-constants
  %(prog)s --model model.xml --output hl.svg --passes highlight -O highlight.names=MatMul_1,Add_2
  %(prog)s --list-passes
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model_graph_viewer.py",
        description="Render a neural network model graph as SVG, optionally simplified by passes.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", help="input model file (OpenVINO IR .xml)")
    parser.add_argument("--output", help="output SVG file")
    parser.add_argument("--passes", default="", help="comma-separated passes, applied in the given order")
    parser.add_argument(
        "-O", "--pass-option", action="append", default=[], metavar="PASS.KEY=VALUE",
        help="option for a selected pass, e.g. highlight.names=a,b (repeatable)",
    )
    parser.add_argument("--no-attributes", action="store_true", help="don't show operation attributes in nodes")
    parser.add_argument("--list-passes", action="store_true", help="list available passes and exit")
    return parser


def parse_pass_options(items: list[str]) -> dict[str, dict[str, str]]:
    """['highlight.names=a,b'] -> {'highlight': {'names': 'a,b'}}"""
    options: dict[str, dict[str, str]] = {}
    for item in items:
        key, sep, value = item.partition("=")
        pass_name, dot, option = key.partition(".")
        if not (sep and dot and pass_name and option):
            raise PassConfigError(f"invalid pass option '{item}', expected PASS.KEY=VALUE")
        options.setdefault(pass_name, {})[option] = value
    return options


def build_passes(passes_arg: str, option_items: list[str]) -> list[TransformationPass]:
    names = [name.strip() for name in passes_arg.split(",") if name.strip()]
    options = parse_pass_options(option_items)
    if unused := set(options) - set(names):
        raise PassConfigError(f"options given for pass(es) not in --passes: {', '.join(sorted(unused))}")
    return [create_pass(name, options.get(name)) for name in names]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(format="warning: %(message)s", level=logging.WARNING)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_passes:
        for name, cls in available_passes().items():
            print(f"{name:16} {cls.description}")
        return 0
    if not args.model or not args.output:
        parser.error("--model and --output are required")

    try:
        # Passes are built first, so a typo in --passes fails before the slow model loading.
        passes = build_passes(args.passes, args.pass_option)
        renderer = GraphvizRenderer(show_attributes=not args.no_attributes)
        result = visualize(args.model, args.output, passes, renderer)
    except (ModelLoadError, PassConfigError, RenderError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"wrote {result.output_path}")
    print(f"graph:  {result.nodes_before} nodes, {result.edges_before} edges"
          f" -> {result.nodes_after} nodes, {result.edges_after} edges")
    print(f"load:   {result.load_seconds:.3f} s")
    for name, seconds in result.pass_timings:
        print(f"pass:   {seconds:.3f} s  {name}")
    print(f"render: {result.render_seconds:.3f} s")
    return 0
