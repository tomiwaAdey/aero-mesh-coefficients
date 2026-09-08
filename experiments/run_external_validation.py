from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aero_mesh.avl_reference import run_avl_mesh_reference
from aero_mesh.classical_models import classification_metrics, predict_class_probabilities
from aero_mesh.dataset import rule_based_edge_labels
from aero_mesh.external import load_crm_lifting_surface_artifact, load_onera_m6_artifact
from aero_mesh.reconstruction import (
    ReconstructionResult,
    reconstruct_aero_mesh_from_edges,
    reconstruct_from_adaptive_planform_envelope,
    source_boundary_errors,
)
from aero_mesh.solver import solve_mesh_steady_vlm
from aero_mesh.stl import canonicalize_aircraft_mesh, edge_feature_table, topology_metrics
from aero_mesh.types import SimulationCase


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
AVL = ROOT / "scripts" / "run_avl_3_32.sh"
SEEDS = (13, 29, 47, 61, 79)
GEOMETRIES = (
    {
        "name": "ONERA_M6",
        "path": ROOT / "vendor" / "onera-m6" / "onera-m6-primary.npz",
        "loader": load_onera_m6_artifact,
        "alpha_deg": 3.06,
        "description": "ONERA M6 generated from the Schmitt-Charpin definition",
        "source_url": "https://www.grc.nasa.gov/WWW/wind/valid/m6wing/m6wing.html",
    },
    {
        "name": "NASA_CRM_DPW4",
        "path": ROOT / "vendor" / "nasa-crm-high-speed" / "dpw4-crm-wing-labelled.npz",
        "loader": load_crm_lifting_surface_artifact,
        "alpha_deg": 5.0,
        "description": "isolated NASA DPW4 high-speed CRM lifting surface",
        "source_url": "https://www.aiaa-dpw.org/Workshop4/DPW4-geom.html",
    },
)


def _mesh_key(result: ReconstructionResult, alpha_deg: float) -> str | None:
    if not result.valid or result.mesh is None:
        return None
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(result.mesh.vertices).tobytes())
    digest.update(np.ascontiguousarray(result.mesh.panels).tobytes())
    digest.update(np.asarray([alpha_deg], dtype="<f8").tobytes())
    return digest.hexdigest()


def _aerodynamic_comparison(
    name: str,
    alpha_deg: float,
    result: ReconstructionResult,
    cache: dict[str, dict[str, object]],
) -> dict[str, object]:
    key = _mesh_key(result, alpha_deg)
    if key is None or result.mesh is None or result.geometry is None:
        return {}
    if key in cache:
        return cache[key]
    case = SimulationCase(f"{name}_external", result.geometry, alpha_deg)
    internal = solve_mesh_steady_vlm(result.mesh, case)
    avl = run_avl_mesh_reference(result.mesh, alpha_deg, AVL)
    row = {
        "cl_internal": internal.cl,
        "cdi_internal": internal.cdi,
        "cm_internal": internal.cm,
        "cl_avl": avl.cl,
        "cdi_avl": avl.cdi,
        "cm_avl": avl.cm,
        "cl_code_to_code_abs_difference": abs(internal.cl - avl.cl),
        "cdi_code_to_code_abs_difference": abs(internal.cdi - avl.cdi),
        "cm_code_to_code_abs_difference": abs(internal.cm - avl.cm),
        "internal_runtime_s": internal.runtime_s,
        "avl_version": avl.version,
        "comparison_scope": "matched low-order code-to-code comparison",
    }
    cache[key] = row
    return row


def _evaluate_geometry(spec: dict[str, object], metadata: dict[str, object]):
    source_path = Path(spec["path"])
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    raw = spec["loader"](source_path)
    raw_topology = topology_metrics(raw)
    if raw_topology["nonmanifold_edges"] != 0 or raw_topology["consistently_oriented"] != 1:
        raise RuntimeError(f"{spec['name']} failed topology validation: {raw_topology}")
    if raw_topology["closed"] != 1 and not raw.metadata.get("intentionally_open", False):
        raise RuntimeError(f"{spec['name']} is open without being declared intentionally open")

    canonical, frame = canonicalize_aircraft_mesh(raw)
    edges, features, reference_labels = edge_feature_table(canonical)
    if reference_labels is None:
        raise RuntimeError(f"{spec['name']} has no immutable reference-boundary annotations")
    reference_result = reconstruct_aero_mesh_from_edges(canonical, edges, reference_labels)
    adaptive_result, adaptive_labels, adaptive_margin = reconstruct_from_adaptive_planform_envelope(
        canonical, edges, features
    )
    rule_labels = rule_based_edge_labels(features)
    rule_result = reconstruct_aero_mesh_from_edges(canonical, edges, rule_labels)

    classification_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    aerodynamic_cache: dict[str, dict[str, object]] = {}
    alpha_deg = float(spec["alpha_deg"])

    for seed in SEEDS:
        model_name = str(metadata["selected_feature_models"][str(seed)])
        model_path = OUT / "models" / f"selected-seed-{seed}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing fitted model {model_path}; run the principal experiment first")
        model = joblib.load(model_path)
        learned_labels = np.asarray(model.predict(features), dtype=int)
        learned_probabilities = predict_class_probabilities(model, features)
        learned_result = reconstruct_aero_mesh_from_edges(
            canonical,
            edges,
            learned_labels,
            edge_probabilities=learned_probabilities,
        )
        fallback_used = not learned_result.valid
        hybrid_result = adaptive_result if fallback_used else learned_result
        hybrid_labels = adaptive_labels if fallback_used else learned_labels
        methods = [
            ("selected_classical", learned_labels, learned_result, False),
            ("gated_hybrid", hybrid_labels, hybrid_result, fallback_used),
        ]
        if seed == SEEDS[0]:
            methods = [
                ("deterministic_rule", rule_labels, rule_result, False),
                ("deterministic_fallback", adaptive_labels, adaptive_result, False),
                ("reference_boundary_lattice", reference_labels, reference_result, False),
                *methods,
            ]
        for method, labels, result, used_fallback in methods:
            metrics, _ = classification_metrics(reference_labels, labels)
            learned_path = method in {"selected_classical", "gated_hybrid"}
            identity = {
                "geometry": spec["name"],
                "seed": seed if learned_path else None,
                "method": method,
                "selected_model": model_name if learned_path else None,
            }
            classification_rows.append({**identity, **metrics})
            reconstruction_rows.append(
                {
                    **identity,
                    "valid": int(result.valid),
                    "failure_code": result.failure_code,
                    "gated_hybrid_used_fallback": (
                        int(used_fallback) if method == "gated_hybrid" else None
                    ),
                    **result.diagnostics,
                    **(source_boundary_errors(canonical, result) if result.valid else {}),
                }
            )
            coefficient_rows.append(
                {
                    **identity,
                    "valid": int(result.valid),
                    "failure_code": result.failure_code,
                    "gated_hybrid_used_fallback": (
                        int(used_fallback) if method == "gated_hybrid" else None
                    ),
                    **_aerodynamic_comparison(
                        str(spec["name"]), alpha_deg, result, aerodynamic_cache
                    ),
                }
            )

    source_data = np.load(source_path)
    geometry_metadata = {
        "geometry": spec["name"],
        "description": spec["description"],
        "source_url": spec["source_url"],
        "source_artifact": str(source_path.relative_to(ROOT)),
        "source_artifact_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "geometry_sha256": str(source_data["geometry_sha256"]),
        "reference_boundary_annotation_hash": hashlib.sha256(
            np.ascontiguousarray(source_data["source_edge_pairs"]).tobytes()
            + np.ascontiguousarray(source_data["source_edge_classes"]).tobytes()
        ).hexdigest(),
        "reference_annotation_method": str(
            source_data.get(
                "reference_annotation_method",
                "preserved construction curves",
            )
        ),
        "reference_annotations_are_algorithmic": bool(
            source_data.get("reference_annotations_are_algorithmic", False)
        ),
        "raw_topology": raw_topology,
        "canonical_topology": topology_metrics(canonical),
        "intentionally_open": bool(raw.metadata.get("intentionally_open", False)),
        "canonical_frame_determinant": float(np.linalg.det(frame.axes)),
        "adaptive_envelope_margin": adaptive_margin,
        "reference_boundary_reconstruction_valid": bool(reference_result.valid),
        "seeds": list(SEEDS),
        "alpha_deg": alpha_deg,
        "aerodynamic_scope": "Matched low-order code-to-code comparison only.",
    }
    return classification_rows, reconstruction_rows, coefficient_rows, geometry_metadata


def main() -> None:
    started = time.perf_counter()
    metadata_path = OUT / "experiment_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Run experiments/run_pipeline.py first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    all_classifications = []
    all_reconstructions = []
    all_coefficients = []
    geometries = []
    for spec in GEOMETRIES:
        classifications, reconstructions, coefficients, geometry_metadata = _evaluate_geometry(
            spec, metadata
        )
        all_classifications.extend(classifications)
        all_reconstructions.extend(reconstructions)
        all_coefficients.extend(coefficients)
        geometries.append(geometry_metadata)

    pd.DataFrame(all_classifications).to_csv(
        OUT / "external_geometry_classification.csv", index=False
    )
    pd.DataFrame(all_reconstructions).to_csv(
        OUT / "external_geometry_reconstruction.csv", index=False
    )
    pd.DataFrame(all_coefficients).to_csv(
        OUT / "external_geometry_coefficients.csv", index=False
    )
    suite_metadata = {
        "description": "Independent lifting-surface geometry-transfer evaluation",
        "geometries": geometries,
        "evaluation_paths": [
            "learned-only reconstruction",
            "deterministic adaptive-envelope fallback",
            "validity-gated hybrid",
        ],
        "fallback_labels_are_predictions_not_reference_labels": True,
        "runtime_s": time.perf_counter() - started,
    }
    (OUT / "external_geometry_metadata.json").write_text(
        json.dumps(suite_metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(suite_metadata, indent=2))


if __name__ == "__main__":
    main()
