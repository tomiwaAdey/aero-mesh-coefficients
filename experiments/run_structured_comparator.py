from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aero_mesh.classical_models import classification_metrics, predict_class_probabilities
from aero_mesh.dataset import generate_geometry_cases, split_geometry_cases
from aero_mesh.potts import edge_adjacency, potts_icm
from aero_mesh.reconstruction import reconstruct_aero_mesh_from_edges
from aero_mesh.solver import solve_mesh_steady_vlm, solve_steady_vlm
from aero_mesh.types import SimulationCase


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEEDS = (13, 29, 47, 61, 79)
MODEL_NAMES = ("svm_rbf", "extra_trees")
WEIGHTS = (0.025, 0.05, 0.1, 0.2, 0.4)
TEST_SPLITS = ("iid", "unseen_mesher", "family_shift")
ALPHA_DEG = 5.0
RECONSTRUCTION_N_SPAN = 20
RECONSTRUCTION_N_CHORD = 4


def _identity(case) -> dict[str, object]:
    return {
        "case": case.name,
        "base_geometry_id": case.base_geometry_id,
        "mesh_id": case.mesh_id,
        "family": case.family,
        "mesher": case.mesher,
        "mesh_density": case.mesh_density,
    }


def _evaluate_cases(
    model,
    cases,
    seed: int,
    split: str,
    model_name: str,
    weight: float,
) -> list[dict[str, object]]:
    rows = []
    reference_cache: dict[str, float] = {}
    for case in cases:
        probabilities = predict_class_probabilities(model, case.features)
        if weight == 0.0:
            predictions = np.asarray(model.predict(case.features), dtype=int)
        else:
            predictions = potts_icm(
                -np.log(np.maximum(probabilities, 1e-12)),
                edge_adjacency(case.edges, case.mesh.vertices),
                weight,
            )
        metrics, _ = classification_metrics(case.labels, predictions)
        reconstruction = reconstruct_aero_mesh_from_edges(
            case.mesh,
            case.edges,
            predictions,
            edge_probabilities=probabilities if weight == 0.0 else None,
        )
        if case.base_geometry_id not in reference_cache:
            reference_cache[case.base_geometry_id] = solve_steady_vlm(
                SimulationCase(
                    case.base_geometry_id,
                    replace(
                        case.params,
                        n_span=RECONSTRUCTION_N_SPAN,
                        n_chord=RECONSTRUCTION_N_CHORD,
                    ),
                    ALPHA_DEG,
                )
            ).cl
        cl_abs_error = np.nan
        if reconstruction.valid and reconstruction.mesh is not None and reconstruction.geometry is not None:
            result = solve_mesh_steady_vlm(
                reconstruction.mesh,
                SimulationCase(case.name, reconstruction.geometry, ALPHA_DEG),
            )
            cl_abs_error = abs(result.cl - reference_cache[case.base_geometry_id])
        rows.append(
            {
                "seed": seed,
                "split": split,
                "method": model_name if weight == 0.0 else f"{model_name}_potts",
                "base_model": model_name,
                "pairwise_weight": weight,
                **_identity(case),
                "valid": int(reconstruction.valid),
                "failure_code": reconstruction.failure_code,
                "cl_abs_error": cl_abs_error,
                **metrics,
            }
        )
    return rows


def _selection_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    paired_valid = frame.loc[frame["valid"] == 1, "cl_abs_error"].dropna()
    return {
        "valid_mesh_rate": float(frame["valid"].mean()),
        "conditional_cl_mae": float(paired_valid.mean()) if len(paired_valid) else float("inf"),
        "macro_f1": float(frame["macro_f1"].mean()),
    }


def main() -> None:
    started = time.perf_counter()
    metadata_path = OUT / "experiment_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Run the principal experiment first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selection_rows = []
    test_rows = []
    for seed in SEEDS:
        cases = generate_geometry_cases(
            {"trapezoidal": 55, "elliptical": 41, "cranked": 20}, seed=seed
        )
        splits = split_geometry_cases(cases)
        selected_candidates = []
        loaded_models = {}
        for model_name in MODEL_NAMES:
            model_path = OUT / "models" / f"{model_name}-seed-{seed}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing {model_path}; run the principal experiment first")
            model = joblib.load(model_path)
            loaded_models[model_name] = model
            rankings = []
            for weight in WEIGHTS:
                rows = _evaluate_cases(
                    model,
                    splits["validation"],
                    seed,
                    "validation",
                    model_name,
                    weight,
                )
                summary = _selection_summary(rows)
                candidate = {
                    "seed": seed,
                    "base_model": model_name,
                    "pairwise_weight": weight,
                    **summary,
                }
                selection_rows.append(candidate)
                rankings.append(candidate)
            selected = sorted(
                rankings,
                key=lambda row: (
                    -float(row["valid_mesh_rate"]),
                    float(row["conditional_cl_mae"]),
                    -float(row["macro_f1"]),
                ),
            )[0]
            selected_candidates.append(selected)

        selected_family = sorted(
            selected_candidates,
            key=lambda row: (
                -float(row["valid_mesh_rate"]),
                float(row["conditional_cl_mae"]),
                -float(row["macro_f1"]),
            ),
        )[0]
        main_model_path = OUT / "models" / f"selected-seed-{seed}.joblib"
        if not main_model_path.exists():
            raise FileNotFoundError(f"Missing {main_model_path}; run the principal experiment first")
        main_model = joblib.load(main_model_path)
        for split in TEST_SPLITS:
            test_rows.extend(
                [
                    {**row, "method": "selected_classical"}
                    for row in _evaluate_cases(
                        main_model,
                        splits[split],
                        seed,
                        split,
                        str(metadata["selected_feature_models"][str(seed)]),
                        0.0,
                    )
                ]
            )
            for candidate in selected_candidates:
                model_name = str(candidate["base_model"])
                weight = float(candidate["pairwise_weight"])
                rows = _evaluate_cases(
                    loaded_models[model_name],
                    splits[split],
                    seed,
                    split,
                    model_name,
                    weight,
                )
                test_rows.extend(rows)
                if model_name == selected_family["base_model"]:
                    test_rows.extend(
                        [
                            {
                                **row,
                                "method": "selected_structured_potts",
                                "selected_base_model": model_name,
                            }
                            for row in rows
                        ]
                    )

    frame = pd.DataFrame(test_rows)
    frame.to_csv(OUT / "potts_comparator_by_geometry.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(OUT / "potts_model_selection.csv", index=False)
    summary = (
        frame.groupby(["split", "method"], as_index=False)
        .agg(
            macro_f1=("macro_f1", "mean"),
            f1_leading=("f1_leading", "mean"),
            f1_trailing=("f1_trailing", "mean"),
            f1_tip=("f1_tip", "mean"),
            valid_mesh_rate=("valid", "mean"),
            conditional_cl_mae=("cl_abs_error", "mean"),
            remeshings=("mesh_id", "count"),
            base_geometries=("base_geometry_id", "nunique"),
        )
    )
    summary.to_csv(OUT / "potts_comparator_summary.csv", index=False)
    run_metadata = {
        "seeds": list(SEEDS),
        "models": list(MODEL_NAMES),
        "weights": list(WEIGHTS),
        "selection_data": "validation geometries only",
        "selection_order": [
            "highest valid-mesh rate",
            "lowest paired-valid CL error",
            "highest geometry-level macro F1",
        ],
        "failure_penalty": None,
        "principal_metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
        "runtime_s": time.perf_counter() - started,
    }
    (OUT / "potts_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(json.dumps(run_metadata, indent=2))


if __name__ == "__main__":
    main()
