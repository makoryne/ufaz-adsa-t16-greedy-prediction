"""
Graph Data Structure (Adjacency List)
Project Work 1 - Task T16: Can we predict when greedy will fail?
Authors: Maruf Abdullayev, Muslum Ibadov
"""

from typing import List, Set, Dict


class SimpleGraph:
    """
    An undirected graph represented as an adjacency list.
    Meets Requirement A3: fully self-implemented, no NetworkX dependency.
    """

    def __init__(self):
        # adj maps: node_id -> set of neighbor node_ids
        self.adj: Dict[int, Set[int]] = {}
        # weights maps: node_id -> float weight (for MWIS)
        self.weights: Dict[int, float] = {}

    def add_node(self, u: int, weight: float = 1.0) -> None:
        """Adds a node to the graph if it doesn't already exist. O(1) amortized."""
        if u not in self.adj:
            self.adj[u] = set()
            self.weights[u] = weight

    def add_edge(self, u: int, v: int) -> None:
        """Adds an undirected edge between u and v. Self-loops are ignored. O(1) amortized."""
        if u == v:
            return  # ignore self-loops
        self.add_node(u)
        self.add_node(v)
        self.adj[u].add(v)
        self.adj[v].add(u)

    def neighbors(self, u: int) -> Set[int]:
        """Returns the set of neighbors for a given node. O(1)."""
        return self.adj.get(u, set())

    def degree(self, u: int) -> int:
        """Returns the degree (number of neighbors) of a node. O(1)."""
        return len(self.adj.get(u, set()))

    def nodes(self) -> List[int]:
        """Returns a list of all nodes. O(V)."""
        return list(self.adj.keys())

    def num_nodes(self) -> int:
        """Returns the total number of vertices |V|. O(1)."""
        return len(self.adj)

    def num_edges(self) -> int:
        """Returns the total number of unique undirected edges |E|. O(V)."""
        return sum(len(nbrs) for nbrs in self.adj.values()) // 2

    def density(self) -> float:
        """Computes the edge density: 2|E| / (|V| * (|V| - 1)). O(V)."""
        n = self.num_nodes()
        if n < 2:
            return 0.0
        return (2.0 * self.num_edges()) / (n * (n - 1))

    def local_clustering(self, u: int) -> float:
        """
        Computes the clustering coefficient of node u:
        The fraction of pairs of neighbors of u that are connected to each other.
        """
        nbrs = list(self.neighbors(u))
        k = len(nbrs)
        if k < 2:
            return 0.0
        
        links_between_neighbors = 0
        for i in range(k):
            for j in range(i + 1, k):
                if nbrs[j] in self.adj[nbrs[i]]:
                    links_between_neighbors += 1

        possible_links = k * (k - 1) / 2.0
        return links_between_neighbors / possible_links


def load_c_elegans_graph(filepath: str) -> SimpleGraph:
    """
    Parses a tab/space-separated edge list file (like ssynlinks.txt)
    and returns a SimpleGraph instance.
    """
    g = SimpleGraph()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Ignore empty lines and comments
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    g.add_edge(u, v)
                except ValueError:
                    continue

    # Assign meaningful structural weights to every node for MWIS
    for node in g.nodes():
        c_u = g.local_clustering(node)
        deg_u = g.degree(node)
        # Weight = degree + 10 * local clustering + 1
        g.weights[node] = round(deg_u + (10.0 * c_u) + 1.0, 2)

    return g