from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aero_mesh.geometry import make_wing_params_for_aspect_ratio
from aero_mesh.types import SimulationCase
from aero_mesh.wake import (
    compare_wake_histories,
    ring_dipole_far_field_error,
    simulate_prescribed_free_wake,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIGURES = OUT / "figures"
METHODS = ("panel", "dipole_fixed", "dipole_gradient")
INK = "#111111"
GREY = "#8b8b8b"
LIGHT_GREY = "#c9c9c9"
BLUE = "#3567b7"
BASELINE_N_SPAN = 16
BASELINE_N_CHORD = 1
BASELINE_TIMESTEP_CHORDS = 0.125
HORIZON_CHORDS = 5.0


def _finish_publication_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(width=0.8, length=4)
    axis.grid(axis="y", color="#e8e8e8", linewidth=0.6, zorder=0)


def _run_case(
    aspect_ratio: float,
    timestep_chords: float,
    n_steps: int,
    core_fraction: float,
    n_span: int = BASELINE_N_SPAN,
    n_chord: int = BASELINE_N_CHORD,
) -> tuple[dict[str, object], dict[str, object]]:
    params = make_wing_params_for_aspect_ratio(
        aspect_ratio,
        name=f"wake_AR{aspect_ratio:g}",
        area=16.0,
        n_span=n_span,
        n_chord=n_chord,
    )
    timestep = timestep_chords * params.root_chord
    case = SimulationCase(
        name=params.name,
        geometry=params,
        alpha_deg=5.0,
        velocity=1.0,
        timestep=timestep,
        n_steps=n_steps,
    )
    histories = {
        method: simulate_prescribed_free_wake(case, method, core_fraction)
        for method in METHODS
    }

    comparison_rows: list[dict[str, object]] = []
    for method in METHODS[1:]:
        comparison_rows.append(
            {
                "aspect_ratio": aspect_ratio,
                "method": method,
                "alpha_deg": case.alpha_deg,
                "timestep_chords": timestep_chords,
                "n_steps": n_steps,
                "nondimensional_horizon": timestep_chords * n_steps,
                "core_radius_chords": core_fraction,
                "n_span": n_span,
                "n_chord": n_chord,
                **compare_wake_histories(histories["panel"], histories[method], params.root_chord),
            }
        )

    history_rows: list[dict[str, object]] = []
    for method, history in histories.items():
        baseline_moment = max(history.moment_mean[0], 1e-12)
        for step in range(n_steps):
            history_rows.append(
                {
                    "aspect_ratio": aspect_ratio,
                    "method": method,
                    "step": step + 1,
                    "nondimensional_time": (step + 1) * timestep_chords,
                    "cl": history.cl[step],
                    "wake_downwash_rms": history.downwash_rms[step],
                    "centroid_x_chords": history.centroid[step, 0] / params.root_chord,
                    "centroid_y_chords": history.centroid[step, 1] / params.root_chord,
                    "centroid_z_chords": history.centroid[step, 2] / params.root_chord,
                    "oldest_cohort_moment_relative_change": abs(history.moment_mean[step] - baseline_moment)
                    / baseline_moment,
                    "minimum_ring_or_equivalent_area": history.minimum_area[step],
                    "runtime_s": history.runtime_s,
                    "finite_state": bool(history.finite_state[step]),
                }
            )

    return {
        "params": params,
        "histories": histories,
        "comparison_rows": comparison_rows,
    }, {"history_rows": history_rows}


def _plot_baseline(result: dict[str, object], timestep_chords: float) -> None:
    params = result["params"]
    histories = result["histories"]
    colors = {"panel": BLUE, "dipole_fixed": LIGHT_GREY, "dipole_gradient": GREY}
    linestyles = {"panel": "-", "dipole_fixed": "--", "dipole_gradient": ":"}
    labels = {
        "panel": "Panel wake",
        "dipole_fixed": "Fixed dipole",
        "dipole_gradient": "Gradient dipole",
    }
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for method in METHODS:
        history = histories[method]
        x = np.arange(1, len(history.cl) + 1) * timestep_chords
        ax.plot(
            x,
            history.cl,
            label=labels[method],
            color=colors[method],
            linestyle=linestyles[method],
            linewidth=1.5,
        )
    ax.set_xlabel("Convected distance / root chord")
    ax.set_ylabel(r"Lift coefficient, $C_L$")
    fig.text(0.105, 0.955, "Explicit-wake lift history", ha="left", va="top", fontsize=15)
    fig.text(
        0.105,
        0.895,
        "Aspect ratio 8 · five-root-chord horizon",
        ha="left",
        va="top",
        color="#555555",
        fontsize=9.5,
    )
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=9.5)
    _finish_publication_axis(ax)
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.15, top=0.75)
    fig.savefig(FIGURES / "wake_lift_history.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for method in METHODS:
        history = histories[method]
        centers = history.element_centers / params.root_chord
        ax.scatter(
            centers[:, 0],
            centers[:, 2],
            s=14,
            alpha=0.72,
            label=labels[method],
            color=colors[method],
            edgecolor=INK if method == "panel" else colors[method],
            linewidth=0.35,
        )
    ax.set_xlabel("x / root chord")
    ax.set_ylabel("z / root chord")
    fig.text(0.105, 0.955, "Final wake-element centres", ha="left", va="top", fontsize=15)
    fig.text(
        0.105,
        0.895,
        "Aspect ratio 8 · after five root chords of convection",
        ha="left",
        va="top",
        color="#555555",
        fontsize=9.5,
    )
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=9.5)
    _finish_publication_axis(ax)
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.15, top=0.75)
    fig.savefig(FIGURES / "wake_final_centres.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for method in METHODS:
        history = histories[method]
        x = np.arange(1, len(history.cl) + 1) * timestep_chords
        axes[0].plot(
            x,
            history.cl,
            label=labels[method],
            color=colors[method],
            linestyle=linestyles[method],
            linewidth=1.5,
        )
        centers = history.element_centers / params.root_chord
        axes[1].scatter(
            centers[:, 0],
            centers[:, 2],
            s=13,
            alpha=0.72,
            color=colors[method],
            edgecolor=INK if method == "panel" else colors[method],
            linewidth=0.3,
        )
    axes[0].set_title("Lift history", loc="left", pad=12)
    axes[0].set_xlabel("Convected distance / root chord")
    axes[0].set_ylabel(r"Lift coefficient, $C_L$")
    axes[1].set_title("Final element centres", loc="left", pad=12)
    axes[1].set_xlabel("x / root chord")
    axes[1].set_ylabel("z / root chord")
    for axis in axes:
        _finish_publication_axis(axis)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.88),
        ncol=3,
        fontsize=9.5,
    )
    fig.text(0.065, 0.965, "Panel and point-dipole wake comparison", ha="left", va="top", fontsize=15)
    fig.text(
        0.065,
        0.91,
        "Aspect ratio 8 · identical wing, timestep and circulation procedure",
        ha="left",
        va="top",
        color="#555555",
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.15, top=0.72, wspace=0.30)
    fig.savefig(FIGURES / "wake_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _panel_case(n_span: int, timestep_chords: float) -> dict[str, float | int | str]:
    params = make_wing_params_for_aspect_ratio(
        8.0,
        name=f"panel_convergence_ns{n_span}_dt{timestep_chords:g}",
        area=16.0,
        n_span=n_span,
        n_chord=1,
    )
    n_steps = round(HORIZON_CHORDS / timestep_chords)
    history = simulate_prescribed_free_wake(
        SimulationCase(
            params.name,
            params,
            alpha_deg=5.0,
            velocity=1.0,
            timestep=timestep_chords * params.root_chord,
            n_steps=n_steps,
        ),
        "panel",
        0.08,
    )
    return {
        "n_span": n_span,
        "n_chord": 1,
        "timestep_chords": timestep_chords,
        "n_steps": n_steps,
        "final_cl": float(history.cl[-1]),
        "settled_cl": float(np.mean(history.cl[-min(5, n_steps) :])),
        "final_centroid_x_chords": float(history.centroid[-1, 0] / params.root_chord),
        "final_centroid_z_chords": float(history.centroid[-1, 2] / params.root_chord),
        "finite_fraction": float(np.mean(history.finite_state)),
        "minimum_ring_area": float(np.min(history.minimum_area)),
        "runtime_s": float(history.runtime_s),
    }


def _panel_convergence() -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, float | int | str]] = []
    for n_span in (4, 8, 12, 16):
        rows.append({"sweep": "spanwise_panels", **_panel_case(n_span, BASELINE_TIMESTEP_CHORDS)})
    for timestep in (0.5, 0.25, 0.125, 0.0625):
        rows.append({"sweep": "timestep", **_panel_case(12, timestep)})
    frame = pd.DataFrame(rows)

    spatial = frame[frame["sweep"] == "spanwise_panels"].sort_values("n_span")
    temporal = frame[frame["sweep"] == "timestep"].sort_values("timestep_chords", ascending=False)
    spatial_pair = spatial.tail(2)
    temporal_pair = temporal.tail(2)

    def relative_change(pair: pd.DataFrame, field: str) -> float:
        values = pair[field].to_numpy(dtype=float)
        return float(abs(values[-1] - values[-2]) / max(abs(values[-1]), 1e-12))

    def absolute_change(pair: pd.DataFrame, field: str) -> float:
        values = pair[field].to_numpy(dtype=float)
        return float(abs(values[-1] - values[-2]))

    summary = {
        "spatial_final_cl_relative_change": relative_change(spatial_pair, "final_cl"),
        "spatial_centroid_x_change_chords": absolute_change(spatial_pair, "final_centroid_x_chords"),
        "temporal_final_cl_relative_change": relative_change(temporal_pair, "final_cl"),
        "temporal_centroid_x_change_chords": absolute_change(temporal_pair, "final_centroid_x_chords"),
        "all_states_finite": bool((frame["finite_fraction"] == 1.0).all()),
    }
    summary["established"] = bool(
        summary["all_states_finite"]
        and summary["spatial_final_cl_relative_change"] <= 0.01
        and summary["temporal_final_cl_relative_change"] <= 0.01
        and summary["spatial_centroid_x_change_chords"] <= 0.1
        and summary["temporal_centroid_x_change_chords"] <= 0.1
    )
    return frame, summary


def main() -> None:
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    baseline_ar8 = None

    convergence, convergence_summary = _panel_convergence()
    convergence.to_csv(OUT / "wake_panel_convergence.csv", index=False)
    if not convergence_summary["established"]:
        raise RuntimeError(f"panel-wake convergence gate failed: {convergence_summary}")

    far_field = pd.DataFrame(
        ring_dipole_far_field_error(np.asarray([1.5, 2.0, 4.0, 8.0, 16.0, 32.0]))
    )
    far_field.to_csv(OUT / "wake_ring_dipole_far_field.csv", index=False)
    far_field_errors = far_field["relative_velocity_error"].to_numpy()
    far_field_summary = {
        "monotonic_error_reduction": bool(np.all(np.diff(far_field_errors) < 0.0)),
        "relative_error_at_16_lengths": float(
            far_field.loc[far_field["distance_over_ring_length"] == 16.0, "relative_velocity_error"].iloc[0]
        ),
    }
    far_field_summary["validated"] = bool(
        far_field_summary["monotonic_error_reduction"]
        and far_field_summary["relative_error_at_16_lengths"] < 0.002
    )
    if not far_field_summary["validated"]:
        raise RuntimeError(f"ring-to-dipole far-field gate failed: {far_field_summary}")

    for aspect_ratio in (4.0, 8.0, 12.0):
        result, history = _run_case(
            aspect_ratio,
            BASELINE_TIMESTEP_CHORDS,
            round(HORIZON_CHORDS / BASELINE_TIMESTEP_CHORDS),
            0.08,
        )
        comparison_rows.extend(result["comparison_rows"])
        history_rows.extend(history["history_rows"])
        if aspect_ratio == 8.0:
            baseline_ar8 = result

    sensitivity_rows: list[dict[str, object]] = []
    for timestep_chords in (0.125, 0.25, 0.5):
        n_steps = round(HORIZON_CHORDS / timestep_chords)
        result, _ = _run_case(8.0, timestep_chords, n_steps, 0.08)
        for row in result["comparison_rows"]:
            sensitivity_rows.append({"sensitivity": "timestep", **row})
    for core_fraction in (0.04, 0.08, 0.12):
        result, _ = _run_case(
            8.0,
            BASELINE_TIMESTEP_CHORDS,
            round(HORIZON_CHORDS / BASELINE_TIMESTEP_CHORDS),
            core_fraction,
        )
        for row in result["comparison_rows"]:
            sensitivity_rows.append({"sensitivity": "core_radius", **row})

    if baseline_ar8 is not None:
        _plot_baseline(baseline_ar8, BASELINE_TIMESTEP_CHORDS)

    comparison = pd.DataFrame(comparison_rows)
    history = pd.DataFrame(history_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    comparison.to_csv(OUT / "wake_comparison.csv", index=False)
    history.to_csv(OUT / "wake_time_history.csv", index=False)
    sensitivity.to_csv(OUT / "wake_sensitivity.csv", index=False)

    best_by_case = (
        comparison.sort_values("trajectory_rms_chords")
        .groupby("aspect_ratio", as_index=False)
        .first()[["aspect_ratio", "method", "trajectory_rms_chords", "cl_mean_abs_delta"]]
        .to_dict(orient="records")
    )
    metadata = {
        "description": "Panel-wake versus point-dipole experiment",
        "methods": list(METHODS),
        "baseline_discretization": {
            "n_span": BASELINE_N_SPAN,
            "n_chord": BASELINE_N_CHORD,
            "timestep_chords": BASELINE_TIMESTEP_CHORDS,
            "horizon_chords": HORIZON_CHORDS,
        },
        "panel_convergence": convergence_summary,
        "ring_dipole_far_field": far_field_summary,
        "best_dipole_by_trajectory": best_by_case,
        "platform": platform.platform(),
        "python": sys.version,
        "runtime_s": time.perf_counter() - start,
        "algorithm_scope": "Classical vortex-ring and point-dipole wake methods.",
    }
    (OUT / "wake_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
