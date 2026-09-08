from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aero_mesh.classical_models import build_model, classification_metrics
from aero_mesh.dataset import (
    balanced_case_arrays,
    generate_geometry_cases,
    split_geometry_cases,
)
from aero_mesh.evaluation import (
    hierarchical_mean_interval,
    hierarchical_paired_metric_summary,
)
from aero_mesh.stl import EDGE_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
DATASET_SEEDS = (13, 29, 47, 61, 79)
TEST_SPLITS = ("iid", "unseen_mesher", "family_shift")
BOOTSTRAP_ITERATIONS = 10_000
FEATURE_GROUPS = {
    "global_position": (0, 1, 2),
    "planform_only": (9, 10, 11),
    "global_geometry_without_planform": tuple(range(9))
    + tuple(range(11, len(EDGE_FEATURE_NAMES))),
    "planform_topology": (7, 8, 9, 10, 11)
    + tuple(range(12, len(EDGE_FEATURE_NAMES))),
    "all_features": tuple(range(len(EDGE_FEATURE_NAMES))),
}
CLASSIFICATION_METRICS = ("macro_f1", "f1_leading", "f1_trailing", "f1_tip")
BOUNDARY_METRICS = (
    "leading_boundary_error_root_chords",
    "trailing_boundary_error_root_chords",
    "tip_boundary_error_root_chords",
)


def _identity(case) -> dict[str, object]:
    return {
        "case": case.name,
        "base_geometry_id": case.base_geometry_id,
        "mesh_id": case.mesh_id,
        "family": case.family,
        "mesher": case.mesher,
        "mesh_density": case.mesh_density,
    }


def _metadata() -> dict[str, object]:
    path = OUT / "experiment_metadata.json"
    if not path.exists():
        raise FileNotFoundError("Run experiments/run_pipeline.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_value(mapping: dict[str, object], seed: int):
    return mapping[str(seed)] if str(seed) in mapping else mapping[seed]


def _predict_rows(
    seed: int,
    split: str,
    cases,
    model_name: str,
    model,
    columns: tuple[int, ...],
) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        predictions = np.asarray(model.predict(case.features[:, columns]), dtype=int)
        metrics, _ = classification_metrics(case.labels, predictions)
        rows.append(
            {
                "seed": seed,
                "split": split,
                "method": model_name,
                **_identity(case),
                **metrics,
            }
        )
    return rows


def _ordered_base_ids(cases, seed: int) -> list[str]:
    identifiers = list(dict.fromkeys(case.base_geometry_id for case in cases))
    rng = np.random.default_rng(seed + 87_301)
    return [identifiers[index] for index in rng.permutation(len(identifiers))]


def run_ablations_and_learning_curves(
    metadata: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ablation_rows: list[dict[str, object]] = []
    learning_rows: list[dict[str, object]] = []
    selected_models = metadata["selected_feature_models"]
    selected_parameters = metadata["selected_hyperparameters"]

    for seed in DATASET_SEEDS:
        cases = generate_geometry_cases(
            {"trapezoidal": 55, "elliptical": 41, "cranked": 20}, seed=seed
        )
        splits = split_geometry_cases(cases)
        fitting_cases = splits["train"] + splits["validation"]
        selected_name = str(_seed_value(selected_models, seed))
        seed_parameters = _seed_value(selected_parameters, seed)
        parameters = dict(seed_parameters[selected_name])
        fitting_x, fitting_y = balanced_case_arrays(fitting_cases, seed=seed + 1_000)

        for group_name, columns in FEATURE_GROUPS.items():
            model = build_model(selected_name, seed, parameters)
            model.fit(fitting_x[:, columns], fitting_y)
            for split in TEST_SPLITS:
                ablation_rows.extend(
                    _predict_rows(seed, split, splits[split], group_name, model, columns)
                )

        ordered_ids = _ordered_base_ids(fitting_cases, seed)
        for training_size in (12, 24, 48, len(ordered_ids)):
            selected_ids = set(ordered_ids[:training_size])
            selected_cases = [
                case for case in fitting_cases if case.base_geometry_id in selected_ids
            ]
            curve_x, curve_y = balanced_case_arrays(
                selected_cases, seed=seed + training_size
            )
            model = build_model(selected_name, seed, parameters)
            model.fit(curve_x, curve_y)
            for split in TEST_SPLITS:
                for row in _predict_rows(
                    seed,
                    split,
                    splits[split],
                    "selected_classical",
                    model,
                    FEATURE_GROUPS["all_features"],
                ):
                    learning_rows.append(
                        {
                            "training_base_geometries": training_size,
                            "training_remeshings": len(selected_cases),
                            **row,
                        }
                    )
    return ablation_rows, learning_rows


def _hierarchical_summary_rows(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        for metric in metrics:
            if metric not in group:
                continue
            interval = hierarchical_mean_interval(
                group[metric].to_numpy(dtype=float),
                group["seed"].to_numpy(),
                group["base_geometry_id"].to_numpy(),
                iterations=BOOTSTRAP_ITERATIONS,
                seed=13,
            )
            rows.append({**identity, "metric": metric, **interval})
    return rows


def _paired_comparison(
    frame: pd.DataFrame,
    *,
    split: str,
    metric: str,
    baseline: str,
    candidate: str,
    higher_is_better: bool,
    analysis: str,
) -> dict[str, object]:
    keys = ["seed", "base_geometry_id", "mesh_id"]
    subset = frame[frame["split"] == split]
    left = subset[subset["method"] == baseline][keys + [metric]].rename(
        columns={metric: "baseline_value"}
    )
    right = subset[subset["method"] == candidate][keys + [metric]].rename(
        columns={metric: "candidate_value"}
    )
    paired = left.merge(right, on=keys, validate="one_to_one")
    summary = hierarchical_paired_metric_summary(
        paired["baseline_value"].to_numpy(dtype=float),
        paired["candidate_value"].to_numpy(dtype=float),
        paired["seed"].to_numpy(),
        paired["base_geometry_id"].to_numpy(),
        higher_is_better=higher_is_better,
        iterations=BOOTSTRAP_ITERATIONS,
        seed=13,
    )
    return {
        "analysis": analysis,
        "split": split,
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        **summary,
    }


def build_principal_statistics() -> list[dict[str, object]]:
    classification = pd.read_csv(OUT / "feature_classification.csv")
    reconstruction = pd.read_csv(OUT / "mesh_reconstruction.csv")
    coefficients = pd.read_csv(OUT / "geometry_to_coefficients.csv")
    reconstruction["failure_rate"] = 1.0 - reconstruction["valid"].astype(float)
    rows: list[dict[str, object]] = []

    for split in TEST_SPLITS:
        for baseline in ("deterministic_rule", "adaptive_envelope"):
            for metric in CLASSIFICATION_METRICS:
                rows.append(
                    _paired_comparison(
                        classification,
                        split=split,
                        metric=metric,
                        baseline=baseline,
                        candidate="selected_classical",
                        higher_is_better=True,
                        analysis="edge_classification",
                    )
                )
            rows.append(
                _paired_comparison(
                    reconstruction,
                    split=split,
                    metric="failure_rate",
                    baseline=baseline,
                    candidate="selected_classical",
                    higher_is_better=False,
                    analysis="reconstruction_failure",
                )
            )
            for metric in BOUNDARY_METRICS:
                rows.append(
                    _paired_comparison(
                        reconstruction,
                        split=split,
                        metric=metric,
                        baseline=baseline,
                        candidate="selected_classical",
                        higher_is_better=False,
                        analysis="boundary_reconstruction",
                    )
                )
            for coefficient in ("cl", "cdi", "cm"):
                rows.append(
                    _paired_comparison(
                        coefficients,
                        split=split,
                        metric=f"{coefficient}_abs_error",
                        baseline=baseline,
                        candidate="selected_classical",
                        higher_is_better=False,
                        analysis="paired_valid_aerodynamic_coefficient",
                    )
                )

    potts_path = OUT / "potts_comparator_by_geometry.csv"
    potts_metadata_path = OUT / "potts_metadata.json"
    potts_is_current = False
    if potts_path.exists() and potts_metadata_path.exists():
        potts_metadata = json.loads(potts_metadata_path.read_text(encoding="utf-8"))
        potts_is_current = potts_metadata.get("principal_metadata_sha256") == hashlib.sha256(
            (OUT / "experiment_metadata.json").read_bytes()
        ).hexdigest()
    if potts_is_current:
        potts = pd.read_csv(potts_path)
        potts["failure_rate"] = 1.0 - potts["valid"].astype(float)
        for split in TEST_SPLITS:
            for metric in (*CLASSIFICATION_METRICS, "failure_rate"):
                rows.append(
                    _paired_comparison(
                        potts,
                        split=split,
                        metric=metric,
                        baseline="selected_classical",
                        candidate="selected_structured_potts",
                        higher_is_better=metric != "failure_rate",
                        analysis="structured_comparator",
                    )
                )
            rows.append(
                _paired_comparison(
                    potts,
                    split=split,
                    metric="cl_abs_error",
                    baseline="selected_classical",
                    candidate="selected_structured_potts",
                    higher_is_better=False,
                    analysis="structured_comparator_paired_valid_cl",
                )
            )
    return rows


def _sensitivity_rows(
    classification: pd.DataFrame,
    reconstruction: pd.DataFrame,
    coefficients: pd.DataFrame,
    dimension: str,
) -> list[dict[str, object]]:
    methods = ("deterministic_rule", "adaptive_envelope", "selected_classical")
    classification = classification[classification["method"].isin(methods)].copy()
    reconstruction = reconstruction[reconstruction["method"].isin(methods)].copy()
    coefficients = coefficients[coefficients["method"].isin(methods)].copy()
    reconstruction["failure_rate"] = 1.0 - reconstruction["valid"].astype(float)
    groups = ["split", dimension, "method"]
    rows = _hierarchical_summary_rows(
        classification, groups, CLASSIFICATION_METRICS
    )
    rows.extend(_hierarchical_summary_rows(reconstruction, groups, ("failure_rate",)))
    rows.extend(
        _hierarchical_summary_rows(
            coefficients,
            groups,
            ("cl_abs_error", "cdi_abs_error", "cm_abs_error"),
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-derived",
        action="store_true",
        help=(
            "Reuse the stored per-geometry ablation and learning-curve predictions. "
            "The default remains a complete refit."
        ),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    metadata = _metadata()
    observed_seeds = tuple(int(value) for value in metadata["dataset_seeds"])
    if observed_seeds != DATASET_SEEDS:
        raise RuntimeError(
            f"Expected five-seed run {DATASET_SEEDS}, found {observed_seeds}"
        )

    if args.reuse_derived:
        ablation = pd.read_csv(OUT / "feature_ablation_by_geometry.csv")
        learning = pd.read_csv(OUT / "learning_curve_by_geometry.csv")
        expected_seeds = set(DATASET_SEEDS)
        if set(ablation["seed"].unique()) != expected_seeds:
            raise RuntimeError("stored feature-ablation predictions use different seeds")
        if set(learning["seed"].unique()) != expected_seeds:
            raise RuntimeError("stored learning-curve predictions use different seeds")
    else:
        ablation_rows, learning_rows = run_ablations_and_learning_curves(metadata)
        ablation = pd.DataFrame(ablation_rows)
        learning = pd.DataFrame(learning_rows)
        ablation.to_csv(OUT / "feature_ablation_by_geometry.csv", index=False)
        learning.to_csv(OUT / "learning_curve_by_geometry.csv", index=False)
    pd.DataFrame(
        _hierarchical_summary_rows(
            ablation, ["split", "method"], CLASSIFICATION_METRICS
        )
    ).to_csv(OUT / "feature_ablation_summary.csv", index=False)
    pd.DataFrame(
        _hierarchical_summary_rows(
            learning,
            ["split", "training_base_geometries"],
            CLASSIFICATION_METRICS,
        )
    ).to_csv(OUT / "learning_curve_summary.csv", index=False)

    classification = pd.read_csv(OUT / "feature_classification.csv")
    reconstruction = pd.read_csv(OUT / "mesh_reconstruction.csv")
    coefficients = pd.read_csv(OUT / "geometry_to_coefficients.csv")
    pd.DataFrame(
        _sensitivity_rows(
            classification, reconstruction, coefficients, "mesh_density"
        )
    ).to_csv(OUT / "mesh_density_sensitivity.csv", index=False)
    pd.DataFrame(
        _sensitivity_rows(classification, reconstruction, coefficients, "mesher")
    ).to_csv(OUT / "triangulation_transfer.csv", index=False)

    statistical = build_principal_statistics()
    pd.DataFrame(statistical).to_csv(OUT / "statistical_summary.csv", index=False)
    analysis_metadata = {
        "runtime_s": time.perf_counter() - started,
        "dataset_seeds": list(DATASET_SEEDS),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "hierarchy": ["dataset/model seed", "base geometry", "remeshing"],
        "model_selection_data": "validation geometries only",
        "learning_curve_unit": "unique physical base geometries",
        "coefficient_population": "intersection of paired valid reconstructions",
        "failure_penalty": None,
        "principal_tests": "paired geometry-level hierarchical bootstrap",
        "reused_per_geometry_predictions": bool(args.reuse_derived),
    }
    (OUT / "statistical_analysis_metadata.json").write_text(
        json.dumps(analysis_metadata, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(statistical).to_string(index=False))
    print(json.dumps(analysis_metadata, indent=2))


if __name__ == "__main__":
    main()
