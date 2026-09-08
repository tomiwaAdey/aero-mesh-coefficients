from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aero_mesh.geometry import generate_remeshed_wing_trimesh, make_wing_params_for_aspect_ratio
from aero_mesh.reconstruction import (
    reconstruct_aero_mesh_from_edges,
    reconstruct_from_adaptive_planform_envelope,
)
from aero_mesh.stl import (
    canonicalize_aircraft_mesh,
    edge_feature_table,
    load_stl,
    repair_trimesh,
    topology_metrics,
    write_binary_stl,
)
from aero_mesh.types import TriMesh


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def _scaled_mesh(mesh: TriMesh, scale: float) -> TriMesh:
    metadata = dict(mesh.metadata)
    if "source_curves" in metadata:
        metadata["source_curves"] = {
            "leading_edge": np.asarray(metadata["source_curves"]["leading_edge"]) * scale,
            "trailing_edge": np.asarray(metadata["source_curves"]["trailing_edge"]) * scale,
            "tips": [np.asarray(curve) * scale for curve in metadata["source_curves"]["tips"]],
        }
    if "source_tip_boundary_options" in metadata:
        metadata["source_tip_boundary_options"] = [
            [np.asarray(curve) * scale for curve in options]
            for options in metadata["source_tip_boundary_options"]
        ]
    if "source_root_chord" in metadata:
        metadata["source_root_chord"] = float(metadata["source_root_chord"]) * scale
    return TriMesh(mesh.vertices * scale, mesh.faces.copy(), mesh.labels, metadata)


def _expanded_vertices(mesh: TriMesh) -> TriMesh:
    vertices = mesh.vertices[mesh.faces].reshape((-1, 3))
    faces = np.arange(len(vertices), dtype=int).reshape((-1, 3))
    source_labels = mesh.metadata.get("source_edge_labels", {})
    expanded_labels: dict[tuple[int, int], int] = {}
    for face_index, source_face in enumerate(mesh.faces):
        target_face = faces[face_index]
        for local_start, local_end in ((0, 1), (1, 2), (2, 0)):
            source_edge = tuple(
                sorted((int(source_face[local_start]), int(source_face[local_end])))
            )
            label = int(source_labels.get(source_edge, 0))
            if label:
                target_edge = tuple(
                    sorted((int(target_face[local_start]), int(target_face[local_end])))
                )
                expanded_labels[target_edge] = label
    metadata = dict(mesh.metadata)
    metadata["source_edge_labels"] = expanded_labels
    return TriMesh(vertices, faces, None, metadata)


def _variants() -> list[tuple[str, TriMesh, bool]]:
    params = make_wing_params_for_aspect_ratio(
        8.0,
        name="stl_robustness_wing",
        sweep_deg=14.0,
        taper_ratio=0.55,
        n_span=12,
        n_chord=4,
    )
    baseline = generate_remeshed_wing_trimesh(
        params,
        mesher="delaunay_uniform",
        mesh_density="medium",
        seed=29,
        rotate=True,
    )
    variants: list[tuple[str, TriMesh, bool]] = [("baseline", baseline, False)]
    for scale in (1e-6, 1e6):
        variants.append((f"scale_{scale:g}", _scaled_mesh(baseline, scale), False))
    variants.append(("repeated_vertices", _expanded_vertices(baseline), False))

    reversed_faces = baseline.faces.copy()
    reversed_faces[::3] = reversed_faces[::3][:, [0, 2, 1]]
    variants.append(
        (
            "reversed_facets",
            TriMesh(baseline.vertices.copy(), reversed_faces, baseline.labels, dict(baseline.metadata)),
            False,
        )
    )

    rng = np.random.default_rng(47)
    retained = np.ones(len(baseline.faces), dtype=bool)
    retained[rng.choice(len(baseline.faces), size=max(1, len(baseline.faces) // 50), replace=False)] = False
    variants.append(
        (
            "missing_facets",
            TriMesh(
                baseline.vertices.copy(),
                baseline.faces[retained].copy(),
                baseline.labels,
                dict(baseline.metadata),
            ),
            True,
        )
    )
    variants.append(
        (
            "nonuniform_density",
            generate_remeshed_wing_trimesh(
                params,
                mesher="delaunay_anisotropic",
                mesh_density="fine",
                seed=61,
                rotate=True,
            ),
            False,
        )
    )
    return variants


def _evaluate(name: str, raw: TriMesh, expect_open: bool, parser: str) -> dict[str, object]:
    source_vertices = len(raw.vertices)
    source_faces = len(raw.faces)
    repaired = repair_trimesh(raw, intentionally_open=expect_open)
    canonical, _ = canonicalize_aircraft_mesh(repaired)
    edges, features, labels = edge_feature_table(canonical)
    adaptive, _, margin = reconstruct_from_adaptive_planform_envelope(
        canonical, edges, features
    )
    source_result = (
        reconstruct_aero_mesh_from_edges(canonical, edges, labels)
        if labels is not None and set(np.unique(labels)) >= {0, 1, 2, 3}
        else None
    )
    metrics = topology_metrics(repaired)
    return {
        "variant": name,
        "parser": parser,
        "source_vertices": source_vertices,
        "source_faces": source_faces,
        "processed_vertices": int(metrics["vertices"]),
        "processed_faces": int(metrics["faces"]),
        "removed_collapsed_faces": repaired.metadata.get("removed_collapsed_faces", 0),
        "removed_zero_area_faces": repaired.metadata.get("removed_zero_area_faces", 0),
        "removed_duplicate_faces": repaired.metadata.get("removed_duplicate_faces", 0),
        "boundary_edges": int(metrics["boundary_edges"]),
        "nonmanifold_edges": int(metrics["nonmanifold_edges"]),
        "closed": int(metrics["closed"]),
        "consistently_oriented": int(metrics["consistently_oriented"]),
        "expected_open": int(expect_open),
        "source_labels_available": int(labels is not None),
        "source_label_reconstruction_valid": (
            int(source_result.valid) if source_result is not None else np.nan
        ),
        "source_label_failure_code": (
            source_result.failure_code if source_result is not None else "LABELS_UNAVAILABLE"
        ),
        "adaptive_reconstruction_valid": int(adaptive.valid),
        "adaptive_failure_code": adaptive.failure_code,
        "adaptive_margin": margin,
    }


def main() -> None:
    started = time.perf_counter()
    rows = [_evaluate(name, mesh, expect_open, "in_memory") for name, mesh, expect_open in _variants()]
    baseline = _variants()[0][1]
    with tempfile.TemporaryDirectory(prefix="aero-stl-robustness-") as directory:
        path = write_binary_stl(Path(directory) / "baseline.stl", baseline)
        rows.append(_evaluate("binary_stl_roundtrip", load_stl(path), False, "binary_stl"))

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "stl_robustness.csv", index=False)
    metadata = {
        "scope": "separate corrupted-input robustness suite",
        "included_in_principal_accuracy_benchmark": False,
        "variants": frame["variant"].tolist(),
        "all_nonmanifold_free": bool((frame["nonmanifold_edges"] == 0).all()),
        "all_expected_closed_status": bool(
            ((frame["closed"] == 0) == (frame["expected_open"] == 1)).all()
        ),
        "runtime_s": time.perf_counter() - started,
    }
    (OUT / "stl_robustness_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
