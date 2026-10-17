"""Framework-independent intermediate graph representation (Node, Edge, Graph).

Must not import openvino or graphviz.
"""

from dataclasses import dataclass, field
from itertools import count
from typing import Any

# None inside the tuple = dynamic dimension ([?, 2048] -> (None, 2048)),
# None as the whole shape = unknown rank or missing metadata, () = scalar.
Shape = tuple[int | None, ...] | None


def format_tensor(dtype: str | None, shape: Shape) -> str:
    """Human-readable tensor type, e.g. 'f32[?, 2048]'. Unknown parts are shown as '?' / '[...]'."""
    if shape is None:
        dims = "[...]"
    else:
        dims = "[" + ", ".join("?" if dim is None else str(dim) for dim in shape) + "]"
    return f"{dtype or '?'}{dims}"


@dataclass
class Node:
    """One operation. `node_id` is unique within a graph; `name` may repeat."""

    node_id: int
    name: str
    op_type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    # Visualization metadata set by passes and read by renderers, e.g. {"border_color": "red"}.
    viz: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    """A tensor flowing from output `src_port` of `src` to input `dst_port` of `dst`.

    Frozen: changing src/dst in place would leave the Graph's indexes pointing at the
    wrong nodes. To reconnect, remove the edge and add a new one.
    """

    edge_id: int
    src: int
    src_port: int
    dst: int
    dst_port: int
    dtype: str | None = None
    shape: Shape = None


class Graph:
    """Directed multigraph of operations with per-node edge indexes.

    All lookups and modifications cost O(degree of the node), never O(nodes) or O(edges).
    """

    def __init__(self) -> None:
        self._nodes: dict[int, Node] = {}
        self._edges: dict[int, Edge] = {}
        # node_id -> {dst_port: edge_id}. Keyed by port: an input has exactly one producer.
        self._in: dict[int, dict[int, int]] = {}
        # node_id -> ids of outgoing edges. A set: one output may feed many consumers.
        self._out: dict[int, set[int]] = {}
        # Ids are never reused, so an id seen earlier (e.g. "shared: 12") stays unambiguous.
        self._node_ids = count()
        self._edge_ids = count()

    # --- nodes ---------------------------------------------------------------

    def add_node(self, name: str, op_type: str, attributes: dict[str, Any] | None = None) -> Node:
        # Copy, so the caller's dict and the node's dict can't change each other.
        node = Node(next(self._node_ids), name, op_type, dict(attributes or {}))
        self._nodes[node.node_id] = node
        self._in[node.node_id] = {}
        self._out[node.node_id] = set()
        return node

    def remove_node(self, node_id: int) -> None:
        self._require_node(node_id)
        # A set, so a self-loop (in and out at once) is removed only once.
        for edge_id in set(self._in[node_id].values()) | self._out[node_id]:
            self.remove_edge(edge_id)
        del self._nodes[node_id], self._in[node_id], self._out[node_id]

    def get_node(self, node_id: int) -> Node:
        self._require_node(node_id)
        return self._nodes[node_id]

    def nodes(self) -> list[Node]:
        """Snapshot list, so passes may add or remove nodes while iterating over it."""
        return list(self._nodes.values())

    def num_nodes(self) -> int:
        return len(self._nodes)

    # --- edges ---------------------------------------------------------------

    def add_edge(
        self,
        src: int,
        src_port: int,
        dst: int,
        dst_port: int,
        dtype: str | None = None,
        shape: Shape = None,
    ) -> Edge:
        self._require_node(src)
        self._require_node(dst)
        if src_port < 0 or dst_port < 0:
            raise ValueError(f"port indexes must be >= 0, got out {src_port} -> in {dst_port}")
        if dst_port in self._in[dst]:
            raise ValueError(
                f"input port {dst_port} of node {dst} is already connected "
                f"(edge {self._in[dst][dst_port]})"
            )

        edge = Edge(next(self._edge_ids), src, src_port, dst, dst_port, dtype, shape)
        self._edges[edge.edge_id] = edge
        self._in[dst][dst_port] = edge.edge_id
        self._out[src].add(edge.edge_id)
        return edge

    def remove_edge(self, edge_id: int) -> None:
        edge = self.get_edge(edge_id)
        del self._edges[edge_id]
        del self._in[edge.dst][edge.dst_port]
        self._out[edge.src].discard(edge_id)

    def get_edge(self, edge_id: int) -> Edge:
        if edge_id not in self._edges:
            raise KeyError(f"no edge with id {edge_id}")
        return self._edges[edge_id]

    def edges(self) -> list[Edge]:
        """Snapshot list, so passes may add or remove edges while iterating over it."""
        return list(self._edges.values())

    def num_edges(self) -> int:
        return len(self._edges)

    def in_edges(self, node_id: int) -> list[Edge]:
        """Incoming edges sorted by destination port."""
        self._require_node(node_id)
        ports = self._in[node_id]
        return [self._edges[ports[port]] for port in sorted(ports)]

    def out_edges(self, node_id: int) -> list[Edge]:
        """Outgoing edges sorted by source port, then edge id (insertion order)."""
        self._require_node(node_id)
        edges = (self._edges[edge_id] for edge_id in self._out[node_id])
        return sorted(edges, key=lambda e: (e.src_port, e.edge_id))

    # --- helpers -------------------------------------------------------------

    def _require_node(self, node_id: int) -> None:
        if node_id not in self._nodes:
            raise KeyError(f"no node with id {node_id}")
