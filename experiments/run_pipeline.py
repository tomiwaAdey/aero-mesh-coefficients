from __future__ import annotations

from dataclasses import replace
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aero_mesh.avl_reference import run_avl_reference
from aero_mesh.classical_models import (
    MODEL_PARAMETER_GRIDS,
    build_model,
    classification_metrics,
    predict_class_probabilities,
)
from aero_mesh.dataset import (
    balanced_case_arrays,
    generate_geometry_cases,
    planform_envelope_edge_labels,
    rule_based_edge_labels,
    split_geometry_cases,
)
from aero_mesh.geometry import generate_aero_mesh, make_wing_params_for_aspect_ratio
from aero_mesh.reconstruction import (
    ReconstructionResult,
    reconstruct_aero_mesh_from_edges,
    reconstruct_from_adaptive_planform_envelope,
    source_boundary_errors,
)
from aero_mesh.solver import prandtl_lift_coefficient, solve_mesh_steady_vlm, solve_steady_vlm
from aero_mesh.types import SimulationCase


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
AVL = ROOT / "scripts" / "run_avl_3_32.sh"
DATASET_SEEDS = (13, 29, 47, 61, 79)
ALPHA_DEG = 5.0
RECONSTRUCTION_N_SPAN = 20
RECONSTRUCTION_N_CHORD = 4
ASPECT_RATIO_TOLERANCE = 0.02
TEST_SPLITS = ("iid", "unseen_mesher", "family_shift")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _case_identity(case) -> dict[str, object]:
    return {
        "case": case.name,
        "base_geometry_id": case.base_geometry_id,
        "mesh_id": case.mesh_id,
        "family": case.family,
        "mesher": case.mesher,
        "mesh_density": case.mesh_density,
        "remesh_seed": case.remesh_seed,
        "source_kind": case.source_kind,
    }


def _measured_planform_aspect_ratio(case) -> float:
    """Measure aspect ratio independently from the preserved source boundaries."""
    source_curves = case.mesh.metadata.get("source_curves", {})
    leading = np.asarray(source_curves.get("leading_edge"), dtype=float)
    trailing = np.asarray(source_curves.get("trailing_edge"), dtype=float)
    if leading.ndim != 2 or trailing.shape != leading.shape or leading.shape[1] != 3:
        raise ValueError(f"{case.name} has invalid source planform curves")
    order = np.argsort(leading[:, 1])
    span_stations = leading[order, 1]
    projected_chord = np.abs(trailing[order, 0] - leading[order, 0])
    reference_area = float(np.trapezoid(projected_chord, span_stations))
    reference_span = float(np.ptp(span_stations))
    if reference_area <= 0.0 or reference_span <= 0.0:
        raise ValueError(f"{case.name} has a non-positive measured planform")
    return reference_span**2 / reference_area


def _reconstruct_model(model, case) -> tuple[np.ndarray, np.ndarray, ReconstructionResult]:
    predictions = np.asarray(model.predict(case.features), dtype=int)
    probabilities = predict_class_probabilities(model, case.features)
    reconstruction = reconstruct_aero_mesh_from_edges(
        case.mesh,
        case.edges,
        predictions,
        edge_probabilities=probabilities,
    )
    return predictions, probabilities, reconstruction


def _conditional_validation_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    conditional = frame.loc[frame["valid"] == 1, "cl_abs_error"].dropna()
    return {
        "macro_f1": float(frame["macro_f1"].mean()),
        "valid_mesh_rate": float(frame["valid"].mean()),
        "conditional_cl_mae": float(conditional.mean()) if len(conditional) else float("inf"),
        "validation_remeshings": int(len(frame)),
        "validation_base_geometries": int(frame["base_geometry_id"].nunique()),
    }


def _validation_rows(model, cases, seed: int, model_name: str, candidate_index: int):
    rows = []
    reference_cache: dict[str, float] = {}
    for case in cases:
        predictions, _, reconstruction = _reconstruct_model(model, case)
        metrics, _ = classification_metrics(case.labels, predictions)
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
        cl_error = np.nan
        if reconstruction.valid and reconstruction.mesh is not None and reconstruction.geometry is not None:
            prediction = solve_mesh_steady_vlm(
                reconstruction.mesh,
                SimulationCase(case.name, reconstruction.geometry, ALPHA_DEG),
            )
            cl_error = abs(prediction.cl - reference_cache[case.base_geometry_id])
        rows.append(
            {
                "seed": seed,
                "model": model_name,
                "candidate_index": candidate_index,
                **_case_identity(case),
                "macro_f1": metrics["macro_f1"],
                "valid": int(reconstruction.valid),
                "failure_code": reconstruction.failure_code,
                "cl_abs_error": cl_error,
            }
        )
    return rows


def select_hyperparameters(seed: int, train_cases, validation_cases):
    train_x, train_y = balanced_case_arrays(train_cases, seed=seed)
    candidate_rows = []
    case_rows = []
    selected_parameters: dict[str, dict[str, object]] = {}
    for model_name, candidates in MODEL_PARAMETER_GRIDS.items():
        rankings = []
        for candidate_index, parameters in enumerate(candidates):
            model = build_model(model_name, seed, parameters)
            fit_start = time.perf_counter()
            model.fit(train_x, train_y)
            fit_runtime = time.perf_counter() - fit_start
            rows = _validation_rows(model, validation_cases, seed, model_name, candidate_index)
            case_rows.extend(rows)
            summary = _conditional_validation_summary(rows)
            candidate_row = {
                "seed": seed,
                "model": model_name,
                "candidate_index": candidate_index,
                "parameters": json.dumps(parameters, sort_keys=True),
                "fit_runtime_s": fit_runtime,
                **summary,
            }
            candidate_rows.append(candidate_row)
            rankings.append(candidate_row)
        selected = sorted(
            rankings,
            key=lambda row: (
                -float(row["valid_mesh_rate"]),
                float(row["conditional_cl_mae"]),
                -float(row["macro_f1"]),
            ),
        )[0]
        selected_parameters[model_name] = dict(candidates[int(selected["candidate_index"])])

    selected_model = sorted(
        (
            row
            for row in candidate_rows
            if json.loads(row["parameters"]) == selected_parameters[str(row["model"])]
        ),
        key=lambda row: (
            -float(row["valid_mesh_rate"]),
            float(row["conditional_cl_mae"]),
            -float(row["macro_f1"]),
        ),
    )[0]
    return selected_parameters, str(selected_model["model"]), candidate_rows, case_rows


def _evaluate_method(method: str, case, model=None):
    if method == "source_label_reference":
        labels = np.asarray(case.labels, dtype=int)
        return labels, reconstruct_aero_mesh_from_edges(case.mesh, case.edges, labels)
    if method == "deterministic_rule":
        labels = rule_based_edge_labels(case.features)
        return labels, reconstruct_aero_mesh_from_edges(case.mesh, case.edges, labels)
    if method == "adaptive_envelope":
        reconstruction, labels, _ = reconstruct_from_adaptive_planform_envelope(
            case.mesh, case.edges, case.features
        )
        return labels, reconstruction
    if model is None:
        raise ValueError(f"model required for {method}")
    labels, _, reconstruction = _reconstruct_model(model, case)
    return labels, reconstruction


def _method_rows(seed: int, split: str, case, method: str, labels, reconstruction, reference):
    metrics, _ = classification_metrics(case.labels, labels)
    reconstruction_row: dict[str, object] = {
        "seed": seed,
        "split": split,
        "method": method,
        **_case_identity(case),
        "valid": int(reconstruction.valid),
        "failure_code": reconstruction.failure_code,
        "leading_edge_points": reconstruction.leading_edge_points,
        "trailing_edge_points": reconstruction.trailing_edge_points,
        "tip_edge_points": reconstruction.tip_edge_points,
        **metrics,
        **reconstruction.diagnostics,
        **source_boundary_errors(case.mesh, reconstruction),
    }
    coefficient_row: dict[str, object] = {
        "seed": seed,
        "split": split,
        "method": method,
        **_case_identity(case),
        "valid": int(reconstruction.valid),
        "failure_code": reconstruction.failure_code,
        "paired_valid": 0,
        "cl_reference": reference.cl,
        "cdi_reference": reference.cdi,
        "cm_reference": reference.cm,
        "cl": np.nan,
        "cdi": np.nan,
        "cm": np.nan,
        "cl_abs_error": np.nan,
        "cdi_abs_error": np.nan,
        "cm_abs_error": np.nan,
        "runtime_s": np.nan,
    }
    if reconstruction.valid and reconstruction.mesh is not None and reconstruction.geometry is not None:
        result = solve_mesh_steady_vlm(
            reconstruction.mesh,
            SimulationCase(case.name, reconstruction.geometry, ALPHA_DEG),
        )
        coefficient_row.update(
            {
                "paired_valid": 1,
                "cl": result.cl,
                "cdi": result.cdi,
                "cm": result.cm,
                "cl_abs_error": abs(result.cl - reference.cl),
                "cdi_abs_error": abs(result.cdi - reference.cdi),
                "cm_abs_error": abs(result.cm - reference.cm),
                "runtime_s": result.runtime_s,
            }
        )
    return reconstruction_row, coefficient_row


def run_geometry_benchmark():
    dataset_rows = []
    selection_rows = []
    selection_case_rows = []
    classification_rows = []
    reconstruction_rows = []
    coefficient_rows = []
    selected_models: dict[int, str] = {}
    selected_parameters: dict[int, dict[str, dict[str, object]]] = {}
    model_directory = OUT / "models"
    model_directory.mkdir(parents=True, exist_ok=True)

    for seed in DATASET_SEEDS:
        cases = generate_geometry_cases(
            {"trapezoidal": 55, "elliptical": 41, "cranked": 20}, seed=seed
        )
        splits = split_geometry_cases(cases)
        for split, split_cases in splits.items():
            for case in split_cases:
                resulting_aspect_ratio = _measured_planform_aspect_ratio(case)
                dataset_rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        **_case_identity(case),
                        "vertices": len(case.mesh.vertices),
                        "faces": len(case.mesh.faces),
                        "edges": len(case.edges),
                        "requested_aspect_ratio": case.params.aspect_ratio,
                        "resulting_aspect_ratio": resulting_aspect_ratio,
                        "aspect_ratio_abs_error": abs(
                            resulting_aspect_ratio - case.params.aspect_ratio
                        ),
                    }
                )

        parameters, selected_name, candidates, validation_cases = select_hyperparameters(
            seed, splits["train"], splits["validation"]
        )
        selected_models[seed] = selected_name
        selected_parameters[seed] = parameters
        selection_rows.extend(candidates)
        selection_case_rows.extend(validation_cases)

        fit_x, fit_y = balanced_case_arrays(
            splits["train"] + splits["validation"], seed=seed + 1000
        )
        fitted = {}
        for model_name, model_parameters in parameters.items():
            fitted[model_name] = build_model(model_name, seed, model_parameters).fit(fit_x, fit_y)
            joblib.dump(
                fitted[model_name],
                model_directory / f"{model_name}-seed-{seed}.joblib",
            )
        joblib.dump(fitted[selected_name], model_directory / f"selected-seed-{seed}.joblib")

        reference_cache = {}
        methods = (
            "source_label_reference",
            "deterministic_rule",
            "adaptive_envelope",
            *fitted.keys(),
        )
        for split in TEST_SPLITS:
            for case in splits[split]:
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
                    )
                for method in methods:
                    labels, reconstruction = _evaluate_method(method, case, fitted.get(method))
                    metrics, _ = classification_metrics(case.labels, labels)
                    classification_rows.append(
                        {
                            "seed": seed,
                            "split": split,
                            "method": method,
                            "selected_model": selected_name,
                            **_case_identity(case),
                            **metrics,
                        }
                    )
                    reconstruction_row, coefficient_row = _method_rows(
                        seed,
                        split,
                        case,
                        method,
                        labels,
                        reconstruction,
                        reference_cache[case.base_geometry_id],
                    )
                    reconstruction_rows.append(reconstruction_row)
                    coefficient_rows.append(coefficient_row)
                    if method == selected_name:
                        classification_rows.append(
                            {**classification_rows[-1], "method": "selected_classical"}
                        )
                        reconstruction_rows.append(
                            {**reconstruction_row, "method": "selected_classical"}
                        )
                        coefficient_rows.append(
                            {**coefficient_row, "method": "selected_classical"}
                        )

    maximum_aspect_ratio_error = max(
        float(row["aspect_ratio_abs_error"]) for row in dataset_rows
    )
    if maximum_aspect_ratio_error > ASPECT_RATIO_TOLERANCE:
        raise RuntimeError(
            "generated planform aspect-ratio mismatch: "
            f"{maximum_aspect_ratio_error:.6f} > {ASPECT_RATIO_TOLERANCE:.6f}"
        )
    return {
        "dataset": dataset_rows,
        "model_selection": selection_rows,
        "model_selection_cases": selection_case_rows,
        "classification": classification_rows,
        "reconstruction": reconstruction_rows,
        "coefficients": coefficient_rows,
        "selected_models": selected_models,
        "selected_parameters": selected_parameters,
        "maximum_aspect_ratio_error": maximum_aspect_ratio_error,
    }


def _validation_case(
    sweep: str,
    params,
    alpha: float,
    wake_length_factor: float = 1.0,
) -> dict[str, object]:
    case = SimulationCase(f"{sweep}_{params.name}_{alpha:g}", params, alpha)
    result = solve_steady_vlm(case, wake_length_factor=wake_length_factor)
    avl = run_avl_reference(params, alpha, AVL)
    return {
        "sweep": sweep,
        "case": case.name,
        "aspect_ratio": params.aspect_ratio,
        "alpha_deg": alpha,
        "twist_deg": params.twist_deg,
        "camber_ratio": params.camber_ratio,
        "n_span": params.n_span,
        "n_chord": params.n_chord,
        "wake_length_factor": wake_length_factor,
        "reference_area": generate_aero_mesh(params).reference_area,
        "reference_chord": generate_aero_mesh(params).reference_chord,
        "reference_span": generate_aero_mesh(params).reference_span,
        "cl_internal": result.cl,
        "cl_avl": avl.cl,
        "cl_error_vs_avl": abs(result.cl - avl.cl),
        "cl_prandtl": prandtl_lift_coefficient(alpha, params.aspect_ratio),
        "cl_error_vs_prandtl": abs(
            result.cl - prandtl_lift_coefficient(alpha, params.aspect_ratio)
        ),
        "cdi_internal": result.cdi,
        "cdi_avl": avl.cdi,
        "cdi_error_vs_avl": abs(result.cdi - avl.cdi),
        "cm_internal": result.cm,
        "cm_avl": avl.cm,
        "cm_error_vs_avl": abs(result.cm - avl.cm),
        "runtime_s": result.runtime_s,
        "avl_version": avl.version,
    }


def run_solver_validation():
    rows = []
    for aspect_ratio in (4, 8, 12, 20):
        params = make_wing_params_for_aspect_ratio(
            aspect_ratio,
            name=f"rectangular_AR{aspect_ratio}",
            n_span=20,
            n_chord=4,
        )
        for alpha in (2.0, 5.0, 8.0):
            rows.append(_validation_case("rectangular_validation", params, alpha))

    base = make_wing_params_for_aspect_ratio(8, name="sweep_AR8", n_span=24, n_chord=6)
    for alpha in (-8.0, -4.0, 0.0, 4.0, 8.0):
        rows.append(_validation_case("angle_of_attack", base, alpha))
    for twist in (-4.0, 0.0, 4.0):
        rows.append(_validation_case("twist", replace(base, twist_deg=twist), 4.0))
    for camber in (0.0, 0.02, 0.04):
        rows.append(_validation_case("camber", replace(base, camber_ratio=camber), 4.0))
    for n_span, n_chord in ((8, 2), (12, 3), (20, 4), (32, 6), (40, 8)):
        rows.append(
            _validation_case(
                "panel_density",
                replace(base, n_span=n_span, n_chord=n_chord),
                5.0,
            )
        )
    for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
        rows.append(_validation_case("wake_length", base, 5.0, wake_length_factor=factor))
    return rows


def run_panel_convergence(solver_rows: list[dict[str, object]] | None = None):
    rows = solver_rows if solver_rows is not None else run_solver_validation()
    return [row for row in rows if row["sweep"] == "panel_density"]


def solver_validation_gates(rows: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    principal = frame[frame["sweep"] == "rectangular_validation"]
    thresholds = {"cl": 0.02, "cdi": 0.003, "cm": 0.01}
    metrics = {
        coefficient: float(principal[f"{coefficient}_error_vs_avl"].mean())
        for coefficient in ("cl", "cdi", "cm")
    }
    return {
        "principal_cases": int(len(principal)),
        "thresholds": thresholds,
        "mean_absolute_error": metrics,
        "passed": {
            coefficient: metrics[coefficient] <= thresholds[coefficient]
            for coefficient in thresholds
        },
        "retained_coefficients": [
            coefficient for coefficient in thresholds if metrics[coefficient] <= thresholds[coefficient]
        ],
    }


def make_figures(results, solver_rows):
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    classification = pd.DataFrame(results["classification"])
    methods = ["deterministic_rule", "adaptive_envelope", "selected_classical"]
    splits = list(TEST_SPLITS)
    summary = (
        classification[classification["method"].isin(methods)]
        .groupby(["split", "method"], as_index=False)["macro_f1"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(methods))
    width = 0.24
    colours = ("#315b7d", "#728b52", "#bb633b")
    for split_index, split in enumerate(splits):
        subset = summary[summary["split"] == split].set_index("method")
        values = [subset.loc[method, "macro_f1"] for method in methods]
        ax.bar(x + (split_index - 1) * width, values, width, label=split.replace("_", " "), color=colours[split_index])
    ax.set_xticks(x, [method.replace("_", " ") for method in methods])
    ax.set_ylabel("Geometry-level macro F1")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "feature_classification.png", dpi=180)
    plt.close(fig)

    solver = pd.DataFrame(solver_rows)
    principal = solver[solver["sweep"] == "rectangular_validation"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, coefficient in zip(axes, ("cl", "cdi", "cm")):
        x_values = principal[f"{coefficient}_avl"]
        y_values = principal[f"{coefficient}_internal"]
        axis.scatter(x_values, y_values, color="#315b7d")
        minimum = min(float(x_values.min()), float(y_values.min()))
        maximum = max(float(x_values.max()), float(y_values.max()))
        axis.plot([minimum, maximum], [minimum, maximum], color="black", linewidth=1)
        axis.set_xlabel(f"AVL {coefficient.upper()}")
        axis.set_ylabel(f"Internal {coefficient.upper()}")
    fig.tight_layout()
    fig.savefig(figures / "avl_validation.png", dpi=180)
    plt.close(fig)

    convergence = solver[solver["sweep"] == "panel_density"].sort_values("n_span")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        convergence["n_span"] * convergence["n_chord"],
        convergence["cl_error_vs_avl"],
        marker="o",
    )
    ax.set_xlabel("Aerodynamic panels")
    ax.set_ylabel("Absolute CL error vs AVL")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "panel_convergence_avl.png", dpi=180)
    plt.close(fig)


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if not AVL.exists():
        raise FileNotFoundError("Run scripts/fetch_avl_reference.sh before the independent benchmark")

    results = run_geometry_benchmark()
    solver_rows = run_solver_validation()
    gates = solver_validation_gates(solver_rows)
    _write_csv(OUT / "geometry_dataset_manifest.csv", results["dataset"])
    _write_csv(OUT / "model_selection.csv", results["model_selection"])
    _write_csv(OUT / "model_selection_by_geometry.csv", results["model_selection_cases"])
    _write_csv(OUT / "feature_classification.csv", results["classification"])
    _write_csv(OUT / "mesh_reconstruction.csv", results["reconstruction"])
    _write_csv(OUT / "geometry_to_coefficients.csv", results["coefficients"])
    _write_csv(OUT / "solver_validation.csv", solver_rows)
    _write_csv(OUT / "panel_convergence.csv", run_panel_convergence(solver_rows))
    (OUT / "solver_validation_gates.json").write_text(
        json.dumps(gates, indent=2), encoding="utf-8"
    )
    make_figures(results, solver_rows)

    metadata = {
        "protocol": "computational geometry and classical machine learning",
        "dataset_seeds": DATASET_SEEDS,
        "selected_feature_models": results["selected_models"],
        "selected_hyperparameters": results["selected_parameters"],
        "selection_order": [
            "highest validation mesh-validity rate",
            "lowest paired-valid validation CL error",
            "highest validation geometry-level macro F1",
        ],
        "invalid_failure_penalty": None,
        "aspect_ratio_validation": {
            "maximum_absolute_error": results["maximum_aspect_ratio_error"],
            "tolerance": ASPECT_RATIO_TOLERANCE,
            "passed": results["maximum_aspect_ratio_error"] <= ASPECT_RATIO_TOLERANCE,
        },
        "avl_executable": str(AVL),
        "solver_validation_gates": gates,
        "runtime_s": time.perf_counter() - start,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (OUT / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
