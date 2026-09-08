import numpy as np

from aero_mesh.potts import decision_to_unary, edge_adjacency, potts_icm


def test_decision_to_unary_prefers_largest_score():
    scores = np.array([[0.1, 2.0, -1.0], [3.0, 0.0, 1.0]])
    unary = decision_to_unary(scores)
    assert np.array_equal(np.argmin(unary, axis=1), np.array([1, 0]))


def test_edge_adjacency_requires_shared_vertex_and_alignment():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=float)
    edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=int)
    neighbours = edge_adjacency(edges, vertices, minimum_alignment=0.8)
    assert neighbours[0][0][0] == 1
    assert not neighbours[2]


def test_potts_icm_removes_isolated_weak_label():
    unary = np.array(
        [
            [0.0, 2.0],
            [0.6, 0.5],
            [0.0, 2.0],
        ]
    )
    neighbours = [[(1, 1.0)], [(0, 1.0), (2, 1.0)], [(1, 1.0)]]
    labels = potts_icm(unary, neighbours, pairwise_weight=0.5)
    assert np.array_equal(labels, np.zeros(3, dtype=int))
