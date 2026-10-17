"""Hide Constant nodes and show their dtype/shape on the consuming operations."""

from collections import defaultdict

from graph_viewer.graph import Graph, format_tensor
from graph_viewer.passes.base import TransformationPass
from graph_viewer.passes.registry import register_pass


@register_pass
class HideConstantsPass(TransformationPass):
    """Constant -> consumer edges become consumer attributes like '[1]': 'f32[1024, 768]'.

    The attribute key keeps the consumer's input port. A Constant with several consumers
    adds '(shared: <constant node id>)'. Tensor values are never read.
    """

    name = "hide-constants"
    description = "Hide Constant nodes; show their dtype and shape as attributes of their consumers."

    def apply(self, graph: Graph) -> Graph:
        # consumer node id -> {input port: text}; attributes are added sorted by port at the end.
        hidden: dict[int, dict[int, str]] = defaultdict(dict)

        for node in graph.nodes():
            if node.op_type != "Constant":
                continue
            edges = graph.out_edges(node.node_id)
            shared = len(edges) > 1
            for edge in edges:
                text = format_tensor(edge.dtype, edge.shape)
                if shared:
                    text += f" (shared: {node.node_id})"
                hidden[edge.dst][edge.dst_port] = text
                graph.remove_edge(edge.edge_id)
            # All of the Constant's edges are hidden now, so it has no visible connection left.
            # A Constant without consumers stays: there is nothing to attach its info to.
            if edges:
                graph.remove_node(node.node_id)

        for consumer_id, ports in hidden.items():
            attributes = graph.get_node(consumer_id).attributes
            for port in sorted(ports):
                attributes[f"[{port}]"] = ports[port]
        return graph
