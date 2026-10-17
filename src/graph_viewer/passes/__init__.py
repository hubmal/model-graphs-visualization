"""Transformation passes operating on the intermediate Graph.

Must not import openvino or graphviz.
"""

from graph_viewer.passes.base import PassConfigError, TransformationPass
# Importing the pass modules registers the built-in passes.
from graph_viewer.passes.fold_converts import FoldConstantConvertsPass
from graph_viewer.passes.hide_constants import HideConstantsPass
from graph_viewer.passes.highlight import HighlightPass
from graph_viewer.passes.pipeline import Pipeline
from graph_viewer.passes.registry import available_passes, create_pass, register_pass

__all__ = [
    "FoldConstantConvertsPass",
    "HideConstantsPass",
    "HighlightPass",
    "PassConfigError",
    "Pipeline",
    "TransformationPass",
    "available_passes",
    "create_pass",
    "register_pass",
]
