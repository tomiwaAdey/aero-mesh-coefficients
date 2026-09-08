#!/usr/bin/env python3
"""Extract and label the lifting surface from the November 2008 DPW4 CRM."""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aero_mesh.stl import edge_topology, load_gmsh_v2_surface, planform_coordinates, topology_metrics


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "vendor" / "nasa-crm-high-speed"
DEFAULT_SOURCE = TARGET / "DPW4_wb_no_tail_v03.igs"
DEFAULT_MESH = TARGET / "dpw4-crm-wing-body.msh"
DEFAULT_OUTPUT = TARGET / "dpw4-crm-wing-labelled.npz"
GMSH = ROOT / "scripts" / "run_gmsh_2_7_0.sh"

# DPW4 CAD entities 4/9 and 10/17 are the lower/upper inboard and
# outboard wing skins. The remaining entities close the trailing edge and tip.
CRM_WING_ENTITY_TAGS = {4, 6, 9, 10, 12, 13, 14, 15, 17}


def _geometry_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _mean_curve(points: np.ndarray, coordinate: int, bins: int = 81) -> np.ndarray:
    order = np.argsort(points[:, coordinate])
    points = points[order]
    groups = np.array_split(points, min(bins, len(points)))
    curve = np.asarray([np.mean(group, axis=0) for group in groups if len(group)], dtype=float)
    return curve[np.argsort(curve[:, coordinate])]


def _shortest_path(
    vertices: np.ndarray,
    edges: np.ndarray,
    costs: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(len(vertices))]
    for edge_index, ((left, right), cost) in enumerate(zip(edges, costs)):
        adjacency[int(left)].append((int(right), edge_index, float(cost)))
        adjacency[int(right)].append((int(left), edge_index, float(cost)))
    distance = np.full(len(vertices), np.inf)
    previous_vertex = np.full(len(vertices), -1, dtype=int)
    previous_edge = np.full(len(vertices), -1, dtype=int)
    distance[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance > distance[current]:
            continue
        if current == end:
            break
        for neighbour, edge_index, cost in adjacency[current]:
            candidate = current_distance + cost
            if candidate < distance[neighbour]:
                distance[neighbour] = candidate
                previous_vertex[neighbour] = current
                previous_edge[neighbour] = edge_index
                heapq.heappush(queue, (candidate, neighbour))
    if not np.isfinite(distance[end]):
        raise RuntimeError("CRM reference boundary is not connected")
    path = []
    current = end
    while current != start:
        path.append(int(previous_edge[current]))
        current = int(previous_vertex[current])
    return np.asarray(path[::-1], dtype=int)


def reference_boundaries(
    vertices: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Construct fixed reference paths on the isolated CAD-derived surface."""
    midpoints = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])
    chord_fraction, span_fraction = planform_coordinates(vertices, midpoints, n_bins=101)
    vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)

    span_min = float(vertices[:, 1].min())
    span_max = float(vertices[:, 1].max())
    span = max(span_max - span_min, 1e-12)
    root_vertices = np.flatnonzero(vertices[:, 1] <= span_min + 0.004 * span)
    tip_vertices = np.flatnonzero(vertices[:, 1] >= span_max - 0.004 * span)
    leading_root = int(root_vertices[np.argmin(vertices[root_vertices, 0])])
    trailing_root = int(root_vertices[np.argmax(vertices[root_vertices, 0])])
    leading_tip = int(tip_vertices[np.argmin(vertices[tip_vertices, 0])])
    trailing_tip = int(tip_vertices[np.argmax(vertices[tip_vertices, 0])])

    normalized_length = lengths / max(float(np.median(lengths)), 1e-12)
    leading_cost = normalized_length * (1.0 + 80.0 * np.clip(chord_fraction, 0.0, 1.0) ** 2)
    trailing_cost = normalized_length * (1.0 + 80.0 * np.clip(1.0 - chord_fraction, 0.0, 1.0) ** 2)
    tip_cost = normalized_length * (1.0 + 80.0 * np.clip(1.0 - span_fraction, 0.0, 1.0) ** 2)
    leading_path = _shortest_path(vertices, edges, leading_cost, leading_root, leading_tip)
    trailing_path = _shortest_path(vertices, edges, trailing_cost, trailing_root, trailing_tip)
    tip_path = _shortest_path(vertices, edges, tip_cost, leading_tip, trailing_tip)

    labels = np.zeros(len(edges), dtype=int)
    labels[leading_path] = 1
    labels[trailing_path] = 2
    labels[tip_path] = 3

    leading_points = np.unique(vertices[edges[leading_path].reshape(-1)], axis=0)
    trailing_points = np.unique(vertices[edges[trailing_path].reshape(-1)], axis=0)
    tip_points = np.unique(vertices[edges[tip_path].reshape(-1)], axis=0)
    curves = {
        "leading_edge": _mean_curve(leading_points, coordinate=1),
        "trailing_edge": _mean_curve(trailing_points, coordinate=1),
        "tip": _mean_curve(tip_points, coordinate=0),
    }
    return edges[labels > 0], labels[labels > 0], curves


def mesh_iges(source: Path, mesh: Path) -> str:
    version_process = subprocess.run(
        [str(GMSH), "-version"], capture_output=True, text=True, check=True
    )
    version_lines = (version_process.stdout + version_process.stderr).splitlines()
    version = next((line.strip() for line in version_lines if line.strip() == "2.7.0"), "")
    if version != "2.7.0":
        raise RuntimeError(f"Expected Gmsh 2.7.0, found {version}")
    process = subprocess.run(
        [
            str(GMSH),
            str(source),
            "-2",
            "-format",
            "msh2",
            "-clmin",
            "120",
            "-clmax",
            "450",
            "-o",
            str(mesh),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    log = process.stdout + process.stderr
    (mesh.parent / "gmsh-2.7.0.log").write_text(log, encoding="utf-8")
    if process.returncode or not mesh.exists():
        raise RuntimeError(f"Gmsh failed ({process.returncode}):\n{log[-4000:]}")
    return version


def prepare(source: Path, mesh_path: Path, output: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}; run scripts/fetch_nasa_crm_reference.sh")
    version = mesh_iges(source, mesh_path)
    half = load_gmsh_v2_surface(mesh_path, entity_tags=CRM_WING_ENTITY_TAGS)
    edges, _ = edge_topology(half)
    labelled_edges, edge_classes, curves = reference_boundaries(half.vertices, edges)
    root_y = float(half.vertices[:, 1].min())
    root_le = float(curves["leading_edge"][np.argmin(curves["leading_edge"][:, 1]), 0])
    root_te = float(curves["trailing_edge"][np.argmin(curves["trailing_edge"][:, 1]), 0])
    geometry_hash = _geometry_sha256(half.vertices, half.faces, labelled_edges, edge_classes)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices=half.vertices,
        faces=half.faces,
        source_edge_pairs=labelled_edges,
        source_edge_classes=edge_classes,
        source_leading_edge=curves["leading_edge"],
        source_trailing_edge=curves["trailing_edge"],
        source_tip=curves["tip"],
        source_root_y=np.asarray(root_y),
        source_root_chord=np.asarray(root_te - root_le),
        geometry_sha256=np.asarray(geometry_hash),
        gmsh_version=np.asarray(version),
        source_sha256=np.asarray(hashlib.sha256(source.read_bytes()).hexdigest()),
        archive_sha256=np.asarray(
            hashlib.sha256(source.with_suffix(source.suffix + ".gz").read_bytes()).hexdigest()
        ),
        mesh_sha256=np.asarray(hashlib.sha256(mesh_path.read_bytes()).hexdigest()),
        source_url=np.asarray("https://www.aiaa-dpw.org/Workshop4/DPW4-geom.html"),
        entity_tags=np.asarray(sorted(CRM_WING_ENTITY_TAGS), dtype=int),
        reference_annotation_method=np.asarray(
            "root-tip extrema joined by minimum-cost paths biased to local planform envelopes"
        ),
        reference_annotations_are_algorithmic=np.asarray(True),
    )
    metadata = {
        "gmsh_version": version,
        "vertices": len(half.vertices),
        "triangles": len(half.faces),
        "labelled_edges": {
            "leading": int(np.sum(edge_classes == 1)),
            "trailing": int(np.sum(edge_classes == 2)),
            "tip": int(np.sum(edge_classes == 3)),
        },
        "geometry_sha256": geometry_hash,
        "entity_tags": sorted(CRM_WING_ENTITY_TAGS),
        "topology": topology_metrics(half),
        "intentionally_open": True,
        "reference_annotation_method": (
            "root-tip extrema joined by minimum-cost paths biased to local planform envelopes"
        ),
        "reference_annotations_are_algorithmic": True,
        "output": str(output.relative_to(ROOT)),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.source, args.mesh, args.output)


if __name__ == "__main__":
    main()
