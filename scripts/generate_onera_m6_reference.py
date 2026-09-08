#!/usr/bin/env python3
"""Generate a labelled full-span ONERA M6 surface from its published definition."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor" / "onera-m6" / "airfoil.txt"
OUTPUT = ROOT / "vendor" / "onera-m6" / "onera-m6-primary.npz"
SEMI_SPAN = 1.1963
ROOT_CHORD = 0.8059
LEADING_SWEEP_DEG = 30.0
TRAILING_SWEEP_DEG = 15.8


def _source_edge_arrays(
    source_edges: list[tuple[int, int, int]],
    inverse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels: dict[tuple[int, int], int] = {}
    for start, end, edge_class in source_edges:
        edge = tuple(sorted((int(inverse[start]), int(inverse[end]))))
        if edge[0] != edge[1]:
            labels[edge] = max(labels.get(edge, 0), edge_class)
    pairs = np.asarray(sorted(labels), dtype=int)
    classes = np.asarray([labels[tuple(pair)] for pair in pairs], dtype=int)
    return pairs, classes


def generate_surface(
    section: np.ndarray,
    n_span: int = 56,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Scale the symmetric ONERA-D section over the complete swept planform."""
    x_fraction = section[:, 0]
    z_fraction = section[:, 1]
    spanwise = SEMI_SPAN * np.sin(np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_span + 1))

    vertices: list[list[float]] = []
    index: dict[tuple[int, int, int], int] = {}
    for side in (1, -1):
        for i, y in enumerate(spanwise):
            leading = np.tan(np.deg2rad(LEADING_SWEEP_DEG)) * abs(y)
            trailing = ROOT_CHORD + np.tan(np.deg2rad(TRAILING_SWEEP_DEG)) * abs(y)
            chord = trailing - leading
            for j, (xf, zf) in enumerate(zip(x_fraction, z_fraction)):
                index[(side, i, j)] = len(vertices)
                vertices.append([leading + xf * chord, y, side * zf * chord])

    faces: list[list[int]] = []
    source_edges: list[tuple[int, int, int]] = []
    last = len(x_fraction) - 1
    for side in (1, -1):
        for i in range(n_span):
            for j in range(last):
                a = index[(side, i, j)]
                b = index[(side, i + 1, j)]
                c = index[(side, i + 1, j + 1)]
                d = index[(side, i, j + 1)]
                pair = ([a, b, d], [b, c, d]) if (i + j) % 2 else ([a, b, c], [a, c, d])
                faces.extend(pair if side == 1 else [triangle[::-1] for triangle in pair])
        for i in range(n_span):
            source_edges.append((index[(side, i, 0)], index[(side, i + 1, 0)], 1))
            source_edges.append((index[(side, i, last)], index[(side, i + 1, last)], 2))
        for i in (0, n_span):
            for j in range(last):
                source_edges.append((index[(side, i, j)], index[(side, i, j + 1)], 3))

    # Close only the physical perimeter. There is no symmetry-plane cap because
    # this is generated as one full-span surface rather than mirrored half meshes.
    for i in range(n_span):
        for j in (0, last):
            upper_a = index[(1, i, j)]
            upper_b = index[(1, i + 1, j)]
            lower_a = index[(-1, i, j)]
            lower_b = index[(-1, i + 1, j)]
            faces.extend(([upper_a, lower_b, upper_b], [upper_a, lower_a, lower_b]))
    for i in (0, n_span):
        for j in range(last):
            upper_a = index[(1, i, j)]
            upper_b = index[(1, i, j + 1)]
            lower_a = index[(-1, i, j)]
            lower_b = index[(-1, i, j + 1)]
            faces.extend(([upper_a, upper_b, lower_b], [upper_a, lower_b, lower_a]))

    vertex_array = np.asarray(vertices, dtype=float)
    face_array = np.asarray(faces, dtype=int)
    unique_vertices, inverse = np.unique(np.round(vertex_array, 12), axis=0, return_inverse=True)
    face_array = inverse[face_array]
    triangles = unique_vertices[face_array]
    valid = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    ) > 1e-12
    face_array = face_array[valid]
    canonical_faces = np.sort(face_array, axis=1)
    _, unique_face_indices = np.unique(canonical_faces, axis=0, return_index=True)
    face_array = face_array[np.sort(unique_face_indices)]
    source_edge_pairs, source_edge_classes = _source_edge_arrays(source_edges, inverse)

    leading = []
    trailing = []
    for y in spanwise:
        leading_x = np.tan(np.deg2rad(LEADING_SWEEP_DEG)) * abs(y)
        trailing_x = ROOT_CHORD + np.tan(np.deg2rad(TRAILING_SWEEP_DEG)) * abs(y)
        leading.append([leading_x, y, 0.0])
        trailing.append([trailing_x, y, 0.0])
    curves = {
        "leading_edge": np.asarray(leading),
        "trailing_edge": np.asarray(trailing),
        "tip_left": np.column_stack(
            [
                leading[0][0] + x_fraction * (trailing[0][0] - leading[0][0]),
                np.full(len(x_fraction), -SEMI_SPAN),
                np.zeros(len(x_fraction)),
            ]
        ),
        "tip_right": np.column_stack(
            [
                leading[-1][0] + x_fraction * (trailing[-1][0] - leading[-1][0]),
                np.full(len(x_fraction), SEMI_SPAN),
                np.zeros(len(x_fraction)),
            ]
        ),
    }
    return unique_vertices, face_array, source_edge_pairs, source_edge_classes, curves


def geometry_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing {SOURCE}; run scripts/fetch_onera_m6_reference.sh")
    section = np.loadtxt(SOURCE)
    if section.shape != (72, 2):
        raise ValueError(f"Expected 72 ONERA-D section coordinates, found {section.shape}")
    vertices, faces, edge_pairs, edge_classes, curves = generate_surface(section)
    digest = geometry_sha256(vertices, faces, edge_pairs, edge_classes)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        vertices=vertices,
        faces=faces,
        source_edge_pairs=edge_pairs,
        source_edge_classes=edge_classes,
        source_leading_edge=curves["leading_edge"],
        source_trailing_edge=curves["trailing_edge"],
        source_tip_left=curves["tip_left"],
        source_tip_right=curves["tip_right"],
        source_root_chord=np.asarray(ROOT_CHORD),
        geometry_sha256=np.asarray(digest),
        source_sha256=np.asarray(hashlib.sha256(SOURCE.read_bytes()).hexdigest()),
        source_url=np.asarray("https://www.grc.nasa.gov/www/wind/valid/m6wing/airfoil.txt"),
    )
    print(
        {
            "vertices": len(vertices),
            "triangles": len(faces),
            "labelled_edges": len(edge_pairs),
            "geometry_sha256": digest,
            "output": str(OUTPUT),
        }
    )


if __name__ == "__main__":
    main()
