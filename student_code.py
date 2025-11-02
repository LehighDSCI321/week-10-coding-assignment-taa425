"implementation of a versatile  digraph"
from collections import deque

class VersatileDigraph():
    "class for versatile digraph"
    #constructor
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.edge_names= {}

    #add_edge
    def add_edge(self, start_node_id, end_node_id, start_node_value=0,
                 end_node_value=0, edge_name=None, edge_weight=1):
        "adds edge to the graph. if node doesnt exist, adds it to the graph"
        #check if start/end node exists. if not adds them to graph
        if start_node_id not in self.nodes:
            self.add_node(start_node_id, start_node_value)
        if end_node_id not in self.nodes:
            self.add_node(end_node_id, end_node_value)
        # initialize edges and edge_names dictionaries for start_node_id if they don't exist
        if start_node_id not in self.edges:
            self.edges[start_node_id] = {}
        if start_node_id not in self.edge_names:
            self.edge_names[start_node_id] = {}
        # check if edge name is provided and unique
        if edge_name is not None:
            if edge_name in self.edge_names[start_node_id]:
                raise ValueError(
                    f"Edge name '{edge_name}' already exists for start node '{start_node_id}'"
                )
            self.edge_names[start_node_id][edge_name] = end_node_id
        self.edges[start_node_id][end_node_id] = (edge_weight, edge_name)#add edge
    def add_node(self, node_id, node_value=0):#add nodes
        "takse node_id and value. checks if already exists; if not, adds to the dictionary"
        self.nodes[node_id] = node_value
    def get_nodes(self): #get nodes
        "returns the list of nodes  in the graph"
        return self.nodes.keys()
    def get_edge_weight(self, start_node_id, end_node_id):#git_edge_weight
        "returns the weight of an edge between two nodes."
        if (start_node_id in self.edges and
            end_node_id in self.edges[start_node_id]):
            return self.edges[start_node_id][end_node_id][0]
        return None
    def get_node_value(self, node_id):#get nodes value
        "return the value of a node. Node value if node exists, None otherwise"
        return self.nodes.get(node_id, None)
    def print_graph(self):
        "prints sentences describing the nodes and edges of the graph"
        print("Nodes:")
        for node_id, value in self.nodes.items():
            print(f"  Node {node_id} with value {value}")
        print("\nEdges:")
        for start_node, edges_dict in self.edges.items():
            for end_node, (weight, name) in edges_dict.items():
                name_str = f"name {name}" if name is not None else "no name"
                print(f"  Edge from {start_node} to {end_node} with weight {weight} and {name_str}")
    def predecessors(self, node_id):
        "Given a node, return a list of nodes that immediately precede that node"
        if node_id not in self.nodes:#check if node exist
            return []#return empty list
        return [start_node for start_node, edges_dict in self.edges.items()if node_id in edges_dict]
    def successors(self, node_id):
        """Given a node, return a list of nodes that immediately succeed that node"""
        if node_id not in self.nodes or node_id not in self.edges:
            return []
        return list(self.edges[node_id].keys())
    def successor_on_edge(self, node_id, edge_name):
        """Given a node and an edge name, identify the successor of
        the node on the edge with the provided name."""
        if (node_id not in self.nodes or
            node_id not in self.edge_names or
            edge_name not in self.edge_names[node_id]):
            return None
        return self.edge_names[node_id][edge_name]
    def in_degree(self, node_id):
        """Given a node, return the number of edges that lead to that node"""
        return len(self.predecessors(node_id))
    def out_degree(self, node_id):
        """Given a node, return the number of edges that lead from that node"""
        if node_id not in self.nodes or node_id not in self.edges:
            return 0
        return len(self.edges[node_id])
class BinaryGraph(VersatileDigraph):
    "BinaryGraph class that inherits from VersatileDigraph"
    def __init__(self):
        super().__init__()
        self.left_children = {}
        self.right_children = {}
        self.add_node("Root", 0)
    def add_node_left(self, child_id, child_value=0, parent_id=None):
        "adds a new left child node to a specified existing node"
        if parent_id is None:
            parent_id = "Root"
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node '{parent_id}' does not exist")
        # check if parent already has left child
        if parent_id in self.left_children and self.left_children[parent_id] is not None:
            raise ValueError(f"Parent node '{parent_id}' already has a left child")
        self.nodes[child_id] = child_value
        self.left_children[parent_id] = child_id
        self.add_edge(parent_id, child_id, edge_name="left_child")
    def add_node_right(self, child_id, child_value=0, parent_id=None):
        "add a new right child node to a specified existing node"
        if parent_id is None:
            parent_id = "Root"
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node '{parent_id}' does not exist")
        #check if parent already has right child
        if parent_id in self.right_children and self.right_children[parent_id] is not None:
            raise ValueError(f"Parent node '{parent_id}' already has a right child")
        self.nodes[child_id] = child_value
        self.right_children[parent_id] = child_id
        self.add_edge(parent_id, child_id, edge_name="right_child")
    def get_node_left(self, parent_id):
        "returns the ID of the left child node for a specified node"
        if parent_id not in self.nodes:
            raise KeyError(f"Parent node '{parent_id}' does not exist")
        return self.left_children.get(parent_id, None)
    def get_node_right(self, parent_id):
        "returns the ID of the right child node for a specified node"
        if parent_id not in self.nodes:
            raise KeyError(f"Parent node '{parent_id}' does not exist")
        return self.right_children.get(parent_id, None)
    def display_tree_text(self, node_id="Root", prefix="", is_left=True):
        "display the tree in a text format similar to your screenshot"
        if node_id is None:
            return
        value = self.get_node_value(node_id)
        print(prefix + ("└── " if is_left else "┌── ") + str(value))
        new_prefix = prefix + ("    " if is_left else "│   ")
        right_child = self.get_node_right(node_id)
        left_child = self.get_node_left(node_id)
        if right_child:
            self.display_tree_text(right_child, new_prefix, False)
        if left_child:
            self.display_tree_text(left_child, new_prefix, True)
class SortingTree(BinaryGraph):
    """SortingTree class that inherits from BinaryTree for binary search tree operations"""
    def __init__(self, root_value=None):
        super().__init__()
        self.nodes = {}
        self.edges = {}
        self.edge_names = {}
        self.left_children = {}
        self.right_children = {}
        self.root_id = None
        if root_value is not None:
            self.root_id = "Root"
            self.add_node(self.root_id, root_value)
        else:
            # if no root value provided, create default root node with value 0
            self.root_id = "Root"
            self.add_node(self.root_id, 0)
    def insert(self, value, node_id=None):
        """Insert value into binary search tree using recursion"""
        if not self.nodes and self.root_id is None:
            self.root_id = "Root"
            self.add_node(self.root_id, value)
            return
        # start recursion from root if no node_id provided
        if node_id is None:
            node_id = self.root_id
        current_value = self.nodes[node_id]
        if value < current_value:
            # gotot left subtree
            left_child = self.get_node_left(node_id)
            if left_child is None:
                # insert left child
                new_id = str(value)
                self.add_node_left(new_id, value, node_id)
            else:
                # recursively insert in left subtree
                self.insert(value, left_child)
        else:
            # Go to right subtree (value >= current_value)
            right_child = self.get_node_right(node_id)
            if right_child is None:
                # insert right child
                new_id = str(value)
                self.add_node_right(new_id, value, node_id)
            else:
                # recursively insert in right subtree
                self.insert(value, right_child)
    def traverse(self, node_id=None, result=None):
        "use a traversal of the tree to display a sorted list of the values in the tree"
        # perform the traversal and get the sorted list
        if result is None:
            result = []
        if node_id is None:
            if self.root_id is None:
                print([])
                return []
            node_id = self.root_id
        # transverse left subtree
        left_child = self.get_node_left(node_id)
        if left_child is not None:
            self.traverse(left_child, result)
        #visit current node
        result.append(self.nodes[node_id])
        # transverse right subtree
        right_child = self.get_node_right(node_id)
        if right_child is not None:
            self.traverse(right_child, result)
        if node_id == self.root_id or (node_id is None and self.root_id is None):
            output = " ".join(str(x) for x in result) + " "
            print(output, end="")
        return result
class SortableDigraph(VersatileDigraph):
    """sortableDigraph class that inherits from VersatileDigraph and adds topological sorting"""
    def top_sort(self):
        "Returns a ordered list of a DAG"
        # calculate in_degree of nodes
        in_degree = {}
        for node in self.nodes:
            in_degree[node] = 0
        # count indegree of nodes
        for edges_dict in self.edges.values():
            for end_node in edges_dict:
                in_degree[end_node] = in_degree.get(end_node, 0) + 1
        # queue with nodes with 0 indegree
        queue = []
        for node, degree in in_degree.items():
            if degree == 0:
                queue.append(node)
        top_order = []
        while queue:
            # remove node from queue
            current_node = queue.pop(0)
            top_order.append(current_node)
            # decrease in-degree for each neighbor of current node
            if current_node in self.edges:
                for neighbor in self.edges[current_node]:
                    in_degree[neighbor] -= 1
                    # add to queue if indegree 0
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        # checks for cycle
        if len(top_order) != len(self.nodes):
            raise ValueError("Graph contains cycles - topological sort not possible")
        return top_order

class TraversableDigraph(SortableDigraph):
    "inherits from SortableDigraph and adds DFS and BFS traversal methods"
    def dfs(self, start_node_id=None):
        " depth-first search traversal of the digraph"
        if not self.nodes:
            return
        # if no start node specified, use the first node
        if start_node_id is None:
            start_node_id = next(iter(self.nodes.keys()))
        if start_node_id not in self.nodes:
            raise ValueError(f"Start node '{start_node_id}' does not exist")
        visited = set()
        def dfs_recursive(node_id):
            """recursive DFS helper function"""
            if node_id in visited:
                return
            visited.add(node_id)
            # visit all successors
            if node_id in self.edges:
                for successor in self.edges[node_id]:
                    if successor not in visited:
                        yield successor
                        yield from dfs_recursive(successor)
        # sgart DFS from the start node but don't yield it
        return dfs_recursive(start_node_id)
    def bfs(self, start_node_id=None):
        """perform breadth-first search traversal of the digraph using deque for efficiency"""
        if not self.nodes:
            return
        # if no start node specified, use the first node
        if start_node_id is None:
            start_node_id = next(iter(self.nodes.keys()))
        if start_node_id not in self.nodes:
            raise ValueError(f"Start node '{start_node_id}' does not exist")
        visited = set()
        queue = deque()
        # start by adding all direct successors of the start node
        if start_node_id in self.edges:
            for successor in self.edges[start_node_id]:
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)
        # process the queue
        while queue:
            current_node = queue.popleft()
            yield current_node
            # add all unvisited successors to the queue
            if current_node in self.edges:
                for successor in self.edges[current_node]:
                    if successor not in visited:
                        visited.add(successor)
                        queue.append(successor)
class DAG(TraversableDigraph):
    "DAG class that inherits from TraversableDigraph and ensures no cycles are created"
    def add_edge(self, start_node_id, end_node_id, start_node_value=0,
                 end_node_value=0, edge_name=None, edge_weight=1):
        "add edge to the graph, ensuring it doesn't create a cycle"
        # check for cycle
        if self._would_create_cycle(start_node_id, end_node_id):
            raise ValueError(
                f"Adding edge from {start_node_id} to {end_node_id} would create a cycle"
            )
        # if no cycle would be created, call parent's add_edge method
        super().add_edge(start_node_id, end_node_id, start_node_value,
                        end_node_value, edge_name, edge_weight)
    def _would_create_cycle(self, start_node_id, end_node_id):
        """check if adding an edge from start_node_id to end_node_id would create a cycle"""
        # if start and end are the same node, it would create a self-loop cycle
        if start_node_id == end_node_id:
            return True
        return self._has_path(end_node_id, start_node_id)
    def _has_path(self, from_node_id, to_node_id):
        """Check if there's a path from from_node_id to to_node_id using BFS"""
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            return False
        if from_node_id == to_node_id:
            return True
        visited = set()
        queue = deque([from_node_id])
        visited.add(from_node_id)
        while queue:
            current_node = queue.popleft()
            # check if we reached the target node
            if current_node == to_node_id:
                return True
            # add all unvisited successors to the queue
            if current_node in self.edges:
                for successor in self.edges[current_node]:
                    if successor not in visited:
                        visited.add(successor)
                        queue.append(successor)
        return False

