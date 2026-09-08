from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aero_mesh.evaluation import hierarchical_mean_interval


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIGURES = OUT / "figures"
SEEDS = (13, 29, 47, 61, 79)
SPLITS = ("iid", "unseen_mesher", "family_shift")
SPLIT_LABELS = {
    "iid": "IID",
    "unseen_mesher": "Unseen mesher",
    "family_shift": "Cranked-wing family",
}
METHODS = ("deterministic_rule", "adaptive_envelope", "selected_classical")
METHOD_LABELS = {
    "deterministic_rule": "Fixed geometrical rule",
    "adaptive_envelope": "Adaptive envelope",
    "selected_classical": "Selected classifier",
}
COLOURS = {
    "deterministic_rule": "#c9c9c9",
    "adaptive_envelope": "#748b56",
    "selected_classical": "#3567b7",
}


def _read_csv(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_json(name: str) -> dict[str, object]:
    path = OUT / name
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def _interval(frame: pd.DataFrame, metric: str) -> dict[str, float | int]:
    return hierarchical_mean_interval(
        frame[metric].to_numpy(dtype=float),
        frame["seed"].to_numpy(),
        frame["base_geometry_id"].to_numpy(),
        iterations=10_000,
        seed=13,
    )


def _finish_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(width=0.8, length=4)
    axis.grid(axis="y", color="#e8e8e8", linewidth=0.6, zorder=0)


def _plot_classification(classification: pd.DataFrame) -> None:
    x = np.arange(len(SPLITS), dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.4, 4.9))
    for method_index, method in enumerate(METHODS):
        estimates = [
            _interval(
                classification[
                    (classification["split"] == split)
                    & (classification["method"] == method)
                ],
                "macro_f1",
            )
            for split in SPLITS
        ]
        means = np.asarray([estimate["mean"] for estimate in estimates], dtype=float)
        errors = np.asarray(
            [
                means - np.asarray([estimate["ci_low"] for estimate in estimates]),
                np.asarray([estimate["ci_high"] for estimate in estimates]) - means,
            ]
        )
        positions = x + (method_index - 1) * width
        axis.bar(
            positions,
            means,
            width,
            color=COLOURS[method],
            edgecolor="#111111",
            linewidth=0.65,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        axis.errorbar(
            positions,
            means,
            yerr=errors,
            fmt="none",
            ecolor="#111111",
            elinewidth=0.8,
            capsize=2.5,
            zorder=4,
        )
    axis.set_ylabel("Geometry-level macro F1")
    axis.set_ylim(0.0, 1.04)
    axis.set_xticks(x, [SPLIT_LABELS[split] for split in SPLITS])
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.29))
    _finish_axis(axis)
    figure.suptitle("Aerodynamic boundary classification", x=0.10, ha="left", fontsize=15)
    figure.text(
        0.10,
        0.895,
        "Means and 95% hierarchical bootstrap intervals over complete geometries",
        color="#555555",
        fontsize=9.5,
    )
    figure.subplots_adjust(left=0.10, right=0.985, top=0.78, bottom=0.25)
    figure.savefig(FIGURES / "feature_classification.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_reconstruction(reconstruction: pd.DataFrame) -> None:
    selected = reconstruction[reconstruction["method"].isin(METHODS)].copy()
    selected["boundary_mean_error_root_chords"] = selected[
        [
            "leading_boundary_error_root_chords",
            "trailing_boundary_error_root_chords",
            "tip_boundary_error_root_chords",
        ]
    ].mean(axis=1)
    x = np.arange(len(SPLITS), dtype=float)
    width = 0.24
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    for method_index, method in enumerate(METHODS):
        valid_estimates = []
        boundary_estimates = []
        for split in SPLITS:
            group = selected[(selected["split"] == split) & (selected["method"] == method)]
            valid_estimates.append(_interval(group, "valid"))
            boundary_estimates.append(_interval(group, "boundary_mean_error_root_chords"))
        positions = x + (method_index - 1) * width
        for axis, estimates in zip(axes, (valid_estimates, boundary_estimates)):
            means = np.asarray([estimate["mean"] for estimate in estimates], dtype=float)
            errors = np.asarray(
                [
                    means - np.asarray([estimate["ci_low"] for estimate in estimates]),
                    np.asarray([estimate["ci_high"] for estimate in estimates]) - means,
                ]
            )
            axis.bar(
                positions,
                means,
                width,
                color=COLOURS[method],
                edgecolor="#111111",
                linewidth=0.65,
                label=METHOD_LABELS[method],
                zorder=3,
            )
            axis.errorbar(
                positions,
                means,
                yerr=errors,
                fmt="none",
                ecolor="#111111",
                elinewidth=0.8,
                capsize=2.5,
                zorder=4,
            )
    axes[0].set_title("Valid aerodynamic meshes", loc="left", fontsize=12)
    axes[0].set_ylabel("Valid-mesh rate")
    axes[0].set_ylim(0.0, 1.04)
    axes[1].set_title("Boundary position", loc="left", fontsize=12)
    axes[1].set_ylabel("Mean error / root chord")
    for axis in axes:
        axis.set_xticks(x, [SPLIT_LABELS[split] for split in SPLITS], rotation=12)
        _finish_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
    figure.suptitle("Structured mesh reconstruction", x=0.07, ha="left", fontsize=15)
    figure.text(
        0.07,
        0.895,
        "Means and 95% hierarchical bootstrap intervals over complete geometries",
        color="#555555",
        fontsize=9.5,
    )
    figure.subplots_adjust(left=0.07, right=0.99, top=0.76, bottom=0.25, wspace=0.28)
    figure.savefig(FIGURES / "mesh_reconstruction.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_solver(solver: pd.DataFrame) -> None:
    principal = solver[solver["sweep"] == "rectangular_validation"].copy()
    panels = (
        ("cl", r"$C_L$", "Lift coefficient"),
        ("cdi", r"$C_{D_i}$", "Induced-drag coefficient"),
        ("cm", r"$C_M$", "Pitching-moment coefficient"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    for axis, (coefficient, symbol, title) in zip(axes, panels):
        reference = principal[f"{coefficient}_avl"].to_numpy(dtype=float)
        internal = principal[f"{coefficient}_internal"].to_numpy(dtype=float)
        lower = min(float(reference.min()), float(internal.min()))
        upper = max(float(reference.max()), float(internal.max()))
        padding = max(0.07 * (upper - lower), 1e-5)
        bounds = (lower - padding, upper + padding)
        axis.plot(bounds, bounds, color="#111111", linewidth=0.9, zorder=1)
        axis.scatter(
            reference,
            internal,
            s=30,
            color="#3567b7",
            edgecolor="#111111",
            linewidth=0.45,
            zorder=3,
        )
        axis.set_title(title, fontsize=11, loc="left")
        axis.set_xlabel(f"AVL {symbol}")
        axis.set_ylabel(f"Internal {symbol}")
        axis.set_xlim(bounds)
        axis.set_ylim(bounds)
        axis.set_aspect("equal", adjustable="box")
        _finish_axis(axis)
    figure.suptitle("Independent aerodynamic implementation comparison", x=0.075, ha="left", fontsize=15)
    figure.text(
        0.075,
        0.88,
        "Twelve matched rectangular-wing cases",
        color="#555555",
        fontsize=9.5,
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.72, wspace=0.43)
    figure.savefig(FIGURES / "avl_validation.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES / "avl_coefficient_validation.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_panel_convergence(convergence: pd.DataFrame) -> None:
    frame = convergence.sort_values("n_span").copy()
    frame["n_panels"] = frame["n_span"] * frame["n_chord"]
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(
        frame["n_panels"],
        frame["cl_error_vs_avl"],
        color="#3567b7",
        marker="o",
        markeredgecolor="#111111",
        markeredgewidth=0.45,
        linewidth=1.4,
    )
    axis.set_xlabel("Aerodynamic panels")
    axis.set_ylabel(r"Absolute $C_L$ difference from AVL")
    axis.set_yscale("log")
    _finish_axis(axis)
    figure.suptitle("Panel-density sensitivity", x=0.105, ha="left", fontsize=15)
    figure.text(0.105, 0.89, "Aspect ratio 8 at 5 degrees", color="#555555", fontsize=9.5)
    figure.subplots_adjust(left=0.105, right=0.985, top=0.78, bottom=0.15)
    figure.savefig(FIGURES / "panel_convergence_avl.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _summary_by_group(
    frame: pd.DataFrame,
    groups: list[str],
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(groups, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(groups, keys))
        for metric in metrics:
            if metric in group:
                rows.append({**identity, "metric": metric, **_interval(group, metric)})
    return rows


def _claim_gate(
    metadata: dict[str, object],
    manifest: pd.DataFrame,
    classification: pd.DataFrame,
    reconstruction: pd.DataFrame,
    solver_gates: dict[str, object],
    external_reconstruction: pd.DataFrame,
    wake_metadata: dict[str, object],
    robustness_metadata: dict[str, object],
) -> dict[str, object]:
    principal = classification[classification["method"] == "selected_classical"]
    principal_reconstruction = reconstruction[reconstruction["method"] == "selected_classical"]
    unseen = principal[principal["split"] == "unseen_mesher"]
    unseen_reconstruction = principal_reconstruction[
        principal_reconstruction["split"] == "unseen_mesher"
    ]
    source_classification = classification[classification["method"] == "source_label_reference"]
    source_reconstruction = reconstruction[reconstruction["method"] == "source_label_reference"]
    learned_external = external_reconstruction[
        external_reconstruction["method"] == "selected_classical"
    ]
    hybrid_external = external_reconstruction[external_reconstruction["method"] == "gated_hybrid"]
    expected_meshers = {
        "structured_alternating",
        "delaunay_uniform",
        "delaunay_random",
        "delaunay_anisotropic",
    }
    checks = {
        "five_seed_protocol": tuple(metadata["dataset_seeds"]) == SEEDS,
        "all_meshers_present": set(manifest["mesher"]) == expected_meshers,
        "three_density_levels_present": set(manifest["mesh_density"]) == {"coarse", "medium", "fine"},
        "aspect_ratio_gate": bool(metadata["aspect_ratio_validation"]["passed"]),
        "source_label_semantics_exact": bool((source_classification["macro_f1"] == 1.0).all()),
        "source_label_reconstruction_complete": bool((source_reconstruction["valid"] == 1).all()),
        "selected_test_splits_complete": set(principal["split"]) == set(SPLITS),
        "unseen_mesher_macro_f1_at_least_0_90": float(unseen["macro_f1"].mean()) >= 0.90,
        "unseen_mesher_valid_rate_at_least_0_95": float(unseen_reconstruction["valid"].mean()) >= 0.95,
        "solver_coefficients_pass": bool(all(solver_gates["passed"].values())),
        "external_learned_seed_count_complete": bool(
            learned_external.groupby("geometry")["seed"].nunique().eq(len(SEEDS)).all()
        ),
        "external_hybrid_reconstruction_complete": bool(
            hybrid_external["geometry"].nunique() == 2
            and (hybrid_external["valid"] == 1).all()
        ),
        "panel_wake_convergence_established": bool(
            wake_metadata["panel_convergence"]["established"]
        ),
        "ring_dipole_far_field_validated": bool(
            wake_metadata["ring_dipole_far_field"]["validated"]
        ),
        "stl_robustness_topology_passed": bool(
            robustness_metadata["all_nonmanifold_free"]
            and robustness_metadata["all_expected_closed_status"]
        ),
    }
    return {
        **checks,
        "all_required_evidence_complete": bool(all(checks.values())),
        "title_narrowing_required": not bool(
            checks["unseen_mesher_macro_f1_at_least_0_90"]
            and checks["unseen_mesher_valid_rate_at_least_0_95"]
        ),
        "direct_published_same_task_comparator_complete": False,
        "state_of_the_art_claim_ready": False,
        "claim_boundary": (
            "The conclusions are restricted to the generated lifting-surface benchmark, "
            "the independent triangulation test, the held-out cranked-wing family, and "
            "the two geometry-transfer examples."
        ),
    }


def _paper_tables(
    statistics: pd.DataFrame,
    solver: pd.DataFrame,
    external_classification: pd.DataFrame,
    external_reconstruction: pd.DataFrame,
    wake: pd.DataFrame,
) -> str:
    lines = [
        "# Generated Classical Result Tables",
        "",
        "This file is generated from the frozen experiment outputs.",
        "",
        "## Paired principal comparisons",
        "",
        "| Analysis | Split | Metric | Baseline | Candidate | Baseline mean | Candidate mean | Improvement | 95% CI | Pairs |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in statistics.iterrows():
        lines.append(
            f"| {row['analysis']} | {row['split']} | {row['metric']} | {row['baseline']} | "
            f"{row['candidate']} | {row['baseline_mean']:.5f} | {row['candidate_mean']:.5f} | "
            f"{row['improvement']:.5f} | {row['improvement_ci_low']:.5f} to "
            f"{row['improvement_ci_high']:.5f} | {int(row['seed_geometry_pairs'])} |"
        )

    principal = solver[solver["sweep"] == "rectangular_validation"]
    lines.extend(
        [
            "",
            "## Independent aerodynamic implementation comparison",
            "",
            "| Coefficient | Cases | Mean absolute difference | Maximum absolute difference |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for coefficient in ("cl", "cdi", "cm"):
        values = principal[f"{coefficient}_error_vs_avl"]
        lines.append(
            f"| {coefficient.upper()} | {len(principal)} | {values.mean():.5f} | {values.max():.5f} |"
        )

    lines.extend(
        [
            "",
            "## External geometry transfer",
            "",
            "| Geometry | Path | Seeds | Mean macro F1 | Valid meshes |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for geometry in sorted(external_classification["geometry"].unique()):
        for method in ("selected_classical", "gated_hybrid"):
            scores = external_classification[
                (external_classification["geometry"] == geometry)
                & (external_classification["method"] == method)
            ]
            meshes = external_reconstruction[
                (external_reconstruction["geometry"] == geometry)
                & (external_reconstruction["method"] == method)
            ]
            lines.append(
                f"| {geometry} | {method} | {scores['seed'].nunique()} | "
                f"{scores['macro_f1'].mean():.4f} | {int(meshes['valid'].sum())}/{len(meshes)} |"
            )

    lines.extend(
        [
            "",
            "## Point-dipole wake appendix",
            "",
            "| Aspect ratio | Method | Trajectory RMS / root chord | Mean absolute CL difference |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for _, row in wake.iterrows():
        lines.append(
            f"| {row['aspect_ratio']:.0f} | {row['method']} | "
            f"{row['trajectory_rms_chords']:.4f} | {row['cl_mean_abs_delta']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metadata = _read_json("experiment_metadata.json")
    solver_gates = _read_json("solver_validation_gates.json")
    wake_metadata = _read_json("wake_metadata.json")
    robustness_metadata = _read_json("stl_robustness_metadata.json")
    manifest = _read_csv("geometry_dataset_manifest.csv")
    classification = _read_csv("feature_classification.csv")
    reconstruction = _read_csv("mesh_reconstruction.csv")
    coefficients = _read_csv("geometry_to_coefficients.csv")
    statistics = _read_csv("statistical_summary.csv")
    solver = _read_csv("solver_validation.csv")
    convergence = _read_csv("panel_convergence.csv")
    external_classification = _read_csv("external_geometry_classification.csv")
    external_reconstruction = _read_csv("external_geometry_reconstruction.csv")
    external_coefficients = _read_csv("external_geometry_coefficients.csv")
    wake = _read_csv("wake_comparison.csv")

    gate = _claim_gate(
        metadata,
        manifest,
        classification,
        reconstruction,
        solver_gates,
        external_reconstruction,
        wake_metadata,
        robustness_metadata,
    )
    if not gate["all_required_evidence_complete"]:
        failed = [key for key, value in gate.items() if isinstance(value, bool) and not value]
        raise RuntimeError(f"evidence gates failed: {failed}")

    _plot_classification(classification)
    _plot_reconstruction(reconstruction)
    _plot_solver(solver)
    _plot_panel_convergence(convergence)

    reconstruction_with_failure = reconstruction.copy()
    reconstruction_with_failure["failure_rate"] = 1.0 - reconstruction_with_failure["valid"]
    summary = {
        "protocol": metadata["protocol"],
        "dataset_seeds": metadata["dataset_seeds"],
        "selected_feature_models": metadata["selected_feature_models"],
        "dataset": {
            "base_geometries": int(manifest["base_geometry_id"].nunique()),
            "remeshings": int(manifest["mesh_id"].nunique()),
            "meshers": sorted(manifest["mesher"].unique().tolist()),
            "density_levels": sorted(manifest["mesh_density"].unique().tolist()),
            "maximum_aspect_ratio_error": float(manifest["aspect_ratio_abs_error"].max()),
        },
        "classification": _summary_by_group(
            classification[classification["method"].isin(METHODS)],
            ["split", "method"],
            ("macro_f1", "f1_leading", "f1_trailing", "f1_tip"),
        ),
        "reconstruction": _summary_by_group(
            reconstruction_with_failure[reconstruction_with_failure["method"].isin(METHODS)],
            ["split", "method"],
            (
                "failure_rate",
                "leading_boundary_error_root_chords",
                "trailing_boundary_error_root_chords",
                "tip_boundary_error_root_chords",
            ),
        ),
        "paired_valid_coefficients": _summary_by_group(
            coefficients[coefficients["method"].isin(METHODS)],
            ["split", "method"],
            ("cl_abs_error", "cdi_abs_error", "cm_abs_error"),
        ),
        "paired_comparisons": _records(statistics),
        "solver_validation": solver_gates,
        "external_geometry": {
            "classification": _records(external_classification),
            "reconstruction": _records(external_reconstruction),
            "coefficients": _records(external_coefficients),
        },
        "wake_appendix": {
            "comparisons": _records(wake),
            "gates": wake_metadata,
        },
        "stl_robustness": robustness_metadata,
        "claim_gate": gate,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "claim_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    (OUT / "paper_tables.md").write_text(
        _paper_tables(
            statistics,
            solver,
            external_classification,
            external_reconstruction,
            wake,
        ),
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
