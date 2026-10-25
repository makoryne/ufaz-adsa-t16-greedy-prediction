"""
Sanity check test for the custom Graph ADT and dataset loading.
"""

import sys
import os

# Allow importing from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import SimpleGraph, load_c_elegans_graph


def test_hand_crafted_graph():
    """Test on a small triangle graph (3 nodes, 3 edges)."""
    g = SimpleGraph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 1)

    assert g.num_nodes() == 3, f"Expected 3 nodes, got {g.num_nodes()}"
    assert g.num_edges() == 3, f"Expected 3 edges, got {g.num_edges()}"
    assert g.degree(1) == 2, f"Expected degree 2, got {g.degree(1)}"
    assert g.local_clustering(1) == 1.0, "Triangle clustering should be 1.0"
    print("Hand-crafted graph test: PASSED")


def test_real_dataset_loading():
    """Verify loading ssynlinks.txt matches the expected C. elegans size."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "ssynlinks.txt")
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Skipping dataset test.")
        return

    g = load_c_elegans_graph(data_path)
    print(f"Successfully loaded C. elegans from {data_path}:")
    print(f"  Nodes: {g.num_nodes()} (expected ~279)")
    print(f"  Edges: {g.num_edges()} (expected ~1961)")
    print(f"  Density: {g.density():.4f}")
    assert g.num_nodes() == 279, f"Node count mismatch: {g.num_nodes()}"
    print("Real dataset loading test: PASSED")


if __name__ == "__main__":
    test_hand_crafted_graph()
    test_real_dataset_loading()