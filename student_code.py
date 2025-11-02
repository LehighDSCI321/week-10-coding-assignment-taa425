"""Week 10: VersatileDigraph / SortableDigraph / TraversableDigraph / DAG."""

from collections import deque
from typing import Dict, List, Set, Union

Numeric = Union[int, float]


# ===================== VersatileDigraph =====================
class VersatileDigraph:
    """A versatile directed graph with support for nodes and edges."""

    def __init__(self) -> None:
        """Initialize node store and adjacency list."""
        # nodes: node_name -> numeric value
        self.nodes: Dict[str, Numeric] = {}
        # adjacency list: source -> set(destinations)
        self._adj: Dict[str, Set[str]] = {}
        # optional: store edge weights and names if provided
        self._weights: Dict[tuple[str, str], Numeric] = {}
        self._edge_names: Dict[tuple[str, str], str] = {}

    # ---------- Node Operations ----------
    def add_node(self, node_name: str, node_value: Numeric = 0) -> None:
        """Add a node with an optional numeric value."""
        if not isinstance(node_name, str):
            raise TypeError("Node name must be a string.")
        if not isinstance(node_value, (int, float)):
            raise TypeError("Node value must be numeric.")
        if node_name in self.nodes:
            raise ValueError(f"Node '{node_name}' already exists.")
        self.nodes[node_name] = node_value
        self._adj.setdefault(node_name, set())

    def get_node_value(self, node_name: str) -> Numeric:
        """Return the numeric value of the node."""
        if node_name not in self.nodes:
            raise KeyError(f"Node '{node_name}' does not exist.")
        return self.nodes[node_name]

    def get_nodes(self) -> List[str]:
        """Return all node names (sorted for determinism)."""
        return sorted(self.nodes.keys())

    # ---------- Edge Operations ----------
    def add_edge(
        self,
        source: str,
        destination: str,
        edge_name: str = "",
        edge_value: Union[Numeric, None] = None,
        edge_weight: Union[Numeric, None] = None,
    ) -> None:
        """Add a directed edge; accepts Week9-compatible kwargs."""
        if source not in self.nodes:
            raise KeyError(f"Source node '{source}' not found.")
        if destination not in self.nodes:
            raise KeyError(f"Destination node '{destination}' not found.")
        if destination in self._adj[source]:
            raise ValueError(f"Edge from '{source}' to '{destination}' already exists.")

        self._adj[source].add(destination)

        # store weight (if provided) for compatibility
        if edge_weight is not None:
            w = edge_weight
        elif edge_value is not None:
            w = edge_value
        else:
            w = 0
        if not isinstance(w, (int, float)):
            raise TypeError("Edge weight must be numeric.")
        self._weights[(source, destination)] = w

        # store edge name for completeness
        self._edge_names[(source, destination)] = str(edge_name or "")

    def get_edge_weight(self, source: str, destination: str) -> Numeric:
        """Return stored weight of an edge (0 if unset)."""
        return self._weights.get((source, destination), 0)

    # ---------- Graph Queries ----------
    def successors(self, node_name: str) -> List[str]:
        """Return all nodes that the given node points to (sorted)."""
        if node_name not in self.nodes:
            raise KeyError(f"Node '{node_name}' does not exist.")
        return sorted(self._adj.get(node_name, set()))

    def predecessors(self, node_name: str) -> List[str]:
        """Return all nodes that have edges leading to the given node (sorted)."""
        if node_name not in self.nodes:
            raise KeyError(f"Node '{node_name}' does not exist.")
        preds = [u for u, nbrs in self._adj.items() if node_name in nbrs]
        return sorted(preds)


# ===================== SortableDigraph =====================
class SortableDigraph(VersatileDigraph):
    """Directed graph that supports topological sorting."""

    def top_sort(self) -> List[str]:
        """Return a topologically sorted order using Kahn's algorithm."""
        indegree: Dict[str, int] = {node: 0 for node in self.nodes}
        for nbrs in self._adj.values():
            for v in nbrs:
                indegree[v] += 1

        q = deque(sorted([n for n, d in indegree.items() if d == 0]))
        order: List[str] = []

        while q:
            u = q.popleft()
            order.append(u)
            for v in sorted(self._adj.get(u, set())):
                indegree[v] -= 1
                if indegree[v] == 0:
                    q.append(v)

        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle — cannot perform topological sort.")
        return order


# ===================== TraversableDigraph =====================
class TraversableDigraph(SortableDigraph):
    """Adds DFS and BFS traversals that yield the next visited node."""

    def _check_start(self, start: str) -> None:
        """Validate that start exists."""
        if start not in self.nodes:
            raise KeyError(f"Start node '{start}' does not exist.")

    def dfs(self, start: str):
        """Depth-first traversal generator starting at `start`, excluding `start`."""
        self._check_start(start)
        visited: Set[str] = {start}
        # seed stack with start's neighbors (reverse-sorted so pop gives ascending)
        stack: List[str] = list(reversed(sorted(self._adj.get(start, set()))))

        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            yield u
            for v in sorted(self._adj.get(u, set()), reverse=True):
                if v not in visited:
                    stack.append(v)

    def bfs(self, start: str):
        """Breadth-first traversal generator starting at `start`, excluding `start`."""
        self._check_start(start)
        visited: Set[str] = {start}
        q: deque[str] = deque(sorted(self._adj.get(start, set())))

        for v in list(q):
            visited.add(v)

        while q:
            u = q.popleft()
            yield u
            for v in sorted(self._adj.get(u, set())):
                if v not in visited:
                    visited.add(v)
                    q.append(v)


# ===================== DAG =====================
class DAG(TraversableDigraph):
    """A directed acyclic graph: add_edge forbids cycle creation."""

    def add_edge(
        self,
        source: str,
        destination: str,
        edge_name: str = "",
        edge_value: Union[Numeric, None] = None,
        edge_weight: Union[Numeric, None] = None,
    ) -> None:
        """Add an edge only if it will not create a cycle; signature kept compatible."""
        if source == destination:
            raise ValueError("Self-loop would create a cycle in a DAG.")

        # If nodes exist, check reachability to avoid cycles
        if source in self.nodes and destination in self.nodes:
            if self._reachable(destination, source):
                raise ValueError("Adding this edge would create a cycle.")

        # Delegate to parent to actually add and record metadata
        super().add_edge(
            source,
            destination,
            edge_name=edge_name,
            edge_value=edge_value,
            edge_weight=edge_weight,
        )

    def _reachable(self, start: str, target: str) -> bool:
        """Return True if there is a path from start to target."""
        if start == target:
            return True
        seen: Set[str] = set()
        stack: List[str] = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            for v in self._adj.get(u, set()):
                if v == target:
                    return True
                if v not in seen:
                    stack.append(v)
        return False
