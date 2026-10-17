"""Fold 'Constant -> Convert' weight-decompression chains into the Constant."""

from graph_viewer.graph import Graph
from graph_viewer.passes.base import TransformationPass
from graph_viewer.passes.registry import register_pass


@register_pass
class FoldConstantConvertsPass(TransformationPass):
    """Remove Convert nodes whose only input is a Constant and connect the Constant directly.

    Models with fp16/int8 weights store them as Constant(f16) -> Convert(f32) -> MatMul.
    Without this pass, hide-constants can't hide those weights, because the MatMul's input
    comes from a Convert. The new edge keeps the Constant's stored dtype (e.g. f16).
    """

    name = "fold-converts"
    description = "Fold Constant -> Convert chains (compressed weights) into the Constant."

    def apply(self, graph: Graph) -> Graph:
        for node in graph.nodes():
            if node.op_type != "Convert":
                continue
            inputs = graph.in_edges(node.node_id)
            if len(inputs) != 1 or graph.get_node(inputs[0].src).op_type != "Constant":
                continue

            source = inputs[0]
            consumers = graph.out_edges(node.node_id)
            # Removing the Convert first frees the consumers' input ports.
            graph.remove_node(node.node_id)
            for edge in consumers:
                graph.add_edge(
                    source.src, source.src_port, edge.dst, edge.dst_port, dtype=source.dtype, shape=source.shape
                )
        return graph
