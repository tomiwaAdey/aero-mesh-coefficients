from __future__ import annotations

from itertools import combinations

import numpy as np


def edge_adjacency(
    edges: np.ndarray,
    vertices: np.ndarray,
    minimum_alignment: float = 0.65,
) -> list[list[tuple[int, float]]]:
    """Connect incident, directionally compatible mesh edges for Potts smoothing."""
    incident: list[list[int]] = [[] for _ in range(len(vertices))]
    for edge_index, (start, end) in enumerate(edges):
        incident[int(start)].append(edge_index)
        incident[int(end)].append(edge_index)
    vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-12)
    neighbours: list[dict[int, float]] = [dict() for _ in range(len(edges))]
    for edge_indices in incident:
        for left, right in combinations(edge_indices, 2):
            alignment = abs(float(np.dot(vectors[left], vectors[right])))
            if alignment < minimum_alignment:
                continue
            weight = (alignment - minimum_alignment) / max(1.0 - minimum_alignment, 1e-12)
            neighbours[left][right] = max(neighbours[left].get(right, 0.0), weight)
            neighbours[right][left] = max(neighbours[right].get(left, 0.0), weight)
    return [sorted(row.items()) for row in neighbours]


def decision_to_unary(decision: np.ndarray) -> np.ndarray:
    """Convert multiclass SVM decision scores to stable relative unary costs."""
    scores = np.asarray(decision, dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores -= np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.maximum(np.sum(probabilities, axis=1, keepdims=True), 1e-12)
    return -np.log(np.maximum(probabilities, 1e-12))


def potts_icm(
    unary: np.ndarray,
    neighbours: list[list[tuple[int, float]]],
    pairwise_weight: float,
    max_iterations: int = 12,
) -> np.ndarray:
    """Minimise a multiclass Potts energy with deterministic ICM updates."""
    labels = np.argmin(unary, axis=1).astype(int)
    classes = np.arange(unary.shape[1])
    for _ in range(max_iterations):
        changes = 0
        for index, adjacent in enumerate(neighbours):
            cost = unary[index].copy()
            for neighbour, weight in adjacent:
                cost += pairwise_weight * weight * (classes != labels[neighbour])
            updated = int(np.argmin(cost))
            if updated != labels[index]:
                labels[index] = updated
                changes += 1
        if changes == 0:
            break
    return labels


def smooth_svm_decisions(
    decision: np.ndarray,
    edges: np.ndarray,
    vertices: np.ndarray,
    pairwise_weight: float,
    minimum_alignment: float = 0.65,
) -> np.ndarray:
    unary = decision_to_unary(decision)
    neighbours = edge_adjacency(edges, vertices, minimum_alignment)
    return potts_icm(unary, neighbours, pairwise_weight)
