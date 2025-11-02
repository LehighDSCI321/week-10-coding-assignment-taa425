'''Josh Berdon week 10 coding assignment'''
import graphviz
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.io import output_file, save
from collections import deque

class VersatileDigraph:
    '''Implement Versatile Diagraph class'''
    def __init__(self):
        '''store node, edges, and edge names'''
        self.__edge_weights = {}
        self.__node_values = {}
        self.__edge_names = {}
        self.__edge_head = {}
    def add_node(self, node_id, node_value=0):
        '''method for adding a node with default value of 0'''
        if not isinstance(node_value, (int, float)):
            raise TypeError("Node value must be a number (int or float)")
        self.__node_values[node_id] = node_value
        self.__edge_weights[node_id]= {}
        self.__edge_names[node_id]= {}
        self.__edge_head[node_id]= {}
    def add_edge(self, tail, head, **vararg):
        '''add edge with optional node values, weight, and name'''
        if not tail in self.get_nodes():
            self.add_node(tail, vararg.get("start_node_value", 0))
        if not head in self.get_nodes():
            self.add_node(head, vararg.get("end_node_value", 0))
        # Handle both 'weight' and 'edge_weight' parameter names
        edge_weight = vararg.get("weight", vararg.get("edge_weight", 0))
        if edge_weight < 0:
            raise ValueError("Edge weight cannot be negative")
        edge_name = vararg.get("edge_name", tail + " to " + head)
        self.__edge_names[tail][head] = edge_name
        self.__edge_head[tail][head] = head
        self.__edge_weights[tail][head] = edge_weight
    def get_nodes(self):
        '''return list of all nodes'''
        return self.__node_values.keys()
    def get_edge_weight(self, tail, head):
        '''return weight of edges from start to end'''
        if tail not in self.__edge_weights or head not in self.__edge_weights[tail]:
            raise KeyError(f"Edge from '{tail}' to '{head}' does not exist")
        return self.__edge_weights[tail][head]
    def get_node_value(self, node):
        """Return value of a node."""
        if node not in self.__node_values:
            raise KeyError(f"Node '{node}' does not exist")
        return self.__node_values[node]
    def predecessors(self, node):
        """given a node, return a list of nodes that immediately precede that node"""
        if node not in self.__node_values:
            raise KeyError(f"Node '{node}' does not exist")
        return [tail for tail, heads in self.__edge_head.items() if node in heads]
    def successors(self, node):
        """given a node, return a list of nodes that immediately succeed that node"""
        if node not in self.__node_values:
            raise KeyError(f"Node '{node}' does not exist")
        return list(self.__edge_head[node].keys())
    def successor_on_edge(self, node, edge_name):
        """given a node and an edge name, identify the successor of 
        the node on the edge with the provided name"""
        if node not in self.__node_values:
            raise KeyError(f"Node '{node}' does not exist")
        matches = [head for head, name in self.__edge_names[node].items() if name == edge_name]
        return matches[0] if matches else None
    def in_degree(self, node):
        """given a node, return the number of edges that lead to that node"""
        if node not in self.__node_values:
            raise KeyError(f"Node '{node}' does not exist")
        return sum(1 for tail, heads in self.__edge_head.items() if node in heads)
    def out_degree(self, node):
        """return the number of edges that lead from the given node"""
        if node not in self.__node_values:
            raise KeyError(f"Node '{node}' does not exist")
        return len(self.__edge_head[node])
    def print_graph(self):
        """Print sentences describing nodes and edges."""
        for tail in self.get_nodes():
            print("Node '" + str(tail) + "' has a value of "+ str(self.get_node_value(tail)) + ".")
            for head in self.__edge_weights[tail]:
                print("There is an edge from node "+ str(tail) + " to node " + str(head) \
                      + " of weight " + str(self.get_edge_weight(tail, head)) + " and name "\
                        + "'" + self.__edge_names[tail][head] + "'.")
    def plot_graph(self):
        """Create a visualization of the graph using GraphViz."""
        dot = graphviz.Digraph(comment='VersatileDigraph')
        # Add nodes with their values
        for node in self.get_nodes():
            node_value = self.get_node_value(node)
            dot.node(str(node), f"{node}\n(value: {node_value})")
        # Add edges with weights and names
        for tail in self.get_nodes():
            for head in self.__edge_weights[tail]:
                weight = self.get_edge_weight(tail, head)
                edge_name = self.__edge_names[tail][head]
                dot.edge(str(tail), str(head), label=f"{edge_name}\n(weight: {weight})")
        # Render and display the graph
        # Try to use the full path to Graphviz if available
        dot.render('graph_visualization', format='png', cleanup=True)
        return dot
    def plot_edge_weights(self):
        """Create a bar graph showing edge weights using Bokeh."""
        # Collect edge data
        edge_labels = []
        weights = []
        for tail in self.get_nodes():
            for head in self.__edge_weights[tail]:
                edge_name = self.__edge_names[tail][head]
                weight = self.get_edge_weight(tail, head)
                edge_labels.append(f"{tail} â†’ {head}\n({edge_name})")
                weights.append(weight)
        # Create the plot
        p = figure(x_range=edge_labels, title="Edge Weights",
                  x_axis_label="Edges", y_axis_label="Weight",
                  width=800, height=400)
        # Add bars
        p.vbar(x=edge_labels, top=weights, width=0.8,
               fill_color="steelblue", line_color="navy", alpha=0.7)
        # Add hover tool
        hover = HoverTool(tooltips=[("Edge", "@x"), ("Weight", "@top")])
        p.add_tools(hover)
        # Rotate x-axis labels for better readability
        p.xaxis.major_label_orientation = 45
        # Save the plot
        output_file("edge_weights_plot.html")
        save(p)
        return p
class SortableDigraph (VersatileDigraph):
    """Class for sortable digraph."""
    def top_sort(self):
        """Return a topologically sorted list of nodes using Kahn's algorithm."""
        # Step 1: Compute in-degrees
        in_degree = {node: self.in_degree(node) for node in self.get_nodes()}
        # Step 2: Initialize queue with nodes having in-degree 0
        queue = [node for node, degree in in_degree.items() if degree == 0]
        sorted_nodes = []
        # Step 3: Process nodes
        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            for succ in self.successors(node):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        # Step 4: Check for cycle
        if len(sorted_nodes) != len(in_degree):
            raise ValueError("Graph has at least one cycle â€” topological sort not possible.")
        return sorted_nodes

class TraversableDigraph(SortableDigraph):
    """Class that extends SortableDigraph with DFS and BFS traversal methods."""
    def dfs(self, start=None):
        """Perform depth first search traversal of the digraph..
        Returns a list of nodes in DFS order."""
        # If no start node specified, use first node in graph
        if start is None:
            nodes = list(self.get_nodes())
            if not nodes:
                return []
            start = nodes[0]
        if start not in self.get_nodes():
            raise KeyError(f"Node '{start}' does not exist")
        # Initialize visited set
        visited = set()
        result = []
        def dfs_visit(node):
            """Recursive helper function for DFS."""
            if node in visited:
                return
            visited.add(node)
            result.append(node)
            # Visit all successors
            for successor in self.successors(node):
                if successor not in visited:
                    dfs_visit(successor)
        dfs_visit(start)
        # If there are unvisited nodes, visit them (for disconnected graphs)
        for node in self.get_nodes():
            if node not in visited:
                dfs_visit(node)
        return result
    def bfs(self, start=None):
        """Perform breadth-first search traversal of the digraph.
        Yields nodes as they are traversed.        """
        # If no start node specified, use first node in graph
        if start is None:
            nodes = list(self.get_nodes())
            if not nodes:
                return
            start = nodes[0]
        if start not in self.get_nodes():
            raise KeyError(f"Node '{start}' does not exist")
        # Initialize visited set and queue
        visited = set()
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            yield node
            # Add all unvisited successors to queue
            for successor in self.successors(node):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)
        # If there are unvisited nodes, continue from them (for disconnected graphs)
        for node in self.get_nodes():
            if node not in visited:
                queue = deque([node])
                visited.add(node)
                while queue:
                    current = queue.popleft()
                    yield current
                    for successor in self.successors(current):
                        if successor not in visited:
                            visited.add(successor)
                            queue.append(successor)

class DAG(TraversableDigraph):
    """Directed Acyclic Graph class.
    Overrides add_edge to ensure no cycles are created."""
    def add_edge(self, tail, head, **vararg):
        """Add an edge, but only if it doesn't create a cycle.
        Before adding edge (tail -> head), checks if there's already a path from head to tail.
        If such a path exists, raises an exception."""
        # First, check if the edge would create a cycle by checking for path from head to tail
        if tail in self.get_nodes() and head in self.get_nodes():
            # Check if there's a path from head to tail using DFS
            if self._has_path(head, tail):
                raise ValueError(
                    f"Adding edge from '{tail}' to '{head}' would create a cycle. "
                    f"There is already a path from '{head}' to '{tail}'."
                )
        # If no cycle would be created, call parent's add_edge
        super().add_edge(tail, head, **vararg)
    def _has_path(self, start, end):
        """Check if there's a path from start node to end node using DFS.
        Returns True if path exists, False otherwise."""
        if start == end:
            return True
        visited = set()
        def dfs_path(current):
            """Recursive DFS to find path."""
            if current == end:
                return True
            if current in visited:
                return False
            visited.add(current)
            for successor in self.successors(current):
                if dfs_path(successor):
                    return True
            return False
        return dfs_path(start)


