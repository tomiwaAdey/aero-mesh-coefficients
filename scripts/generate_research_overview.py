#!/usr/bin/env python3
"""Generate the repository overview from the study's geometry implementation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from aero_mesh.geometry import (
    LABEL_LEADING_EDGE,
    LABEL_TIP,
    LABEL_TRAILING_EDGE,
    generate_aero_mesh,
    generate_wing_surface,
    make_wing_params_for_aspect_ratio,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures" / "geometry-to-coefficients-overview.png"
INK = "#242424"
MUTED = "#aaa69e"
BLUE = "#1647c8"
RED = "#e44d3a"
GOLD = "#d59412"
PAPER = "#f2f0ea"


def project(points: np.ndarray) -> np.ndarray:
    """Use a fixed oblique projection shared by all three stages."""
    projected = np.empty((len(points), 2), dtype=float)
    projected[:, 0] = points[:, 1]
    projected[:, 1] = 0.62 * points[:, 0] - 1.85 * points[:, 2]
    return projected


def style_axis(axis: plt.Axes) -> None:
    axis.set_aspect("equal")
    axis.set_anchor("N")
    axis.axis("off")


def generate() -> None:
    params = make_wing_params_for_aspect_ratio(
        8.0,
        name="overview-wing",
        area=16.0,
        sweep_deg=17.0,
        taper_ratio=0.48,
        n_span=18,
        n_chord=6,
    )
    vertices, faces, labels = generate_wing_surface(
        params,
        n_span_surface=30,
        n_chord_surface=18,
    )
    points = project(vertices)
    lattice = generate_aero_mesh(params)
    lattice_points = project(lattice.vertices)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=160)
    figure.patch.set_facecolor(PAPER)
    for axis in axes:
        axis.set_facecolor(PAPER)

    style_axis(axes[0])
    triangle_segments = []
    for face in faces[::2]:
        triangle = points[face]
        triangle_segments.extend(
            ([triangle[0], triangle[1]], [triangle[1], triangle[2]], [triangle[2], triangle[0]])
        )
    axes[0].add_collection(
        LineCollection(triangle_segments, colors=MUTED, linewidths=0.22, alpha=0.3)
    )
    axes[0].scatter(points[:, 0], points[:, 1], s=2.0, color=INK, alpha=0.62)

    style_axis(axes[1])
    axes[1].scatter(points[:, 0], points[:, 1], s=1.2, color=MUTED, alpha=0.3)
    for label, colour, size in (
        (LABEL_LEADING_EDGE, BLUE, 7.0),
        (LABEL_TRAILING_EDGE, RED, 7.0),
        (LABEL_TIP, GOLD, 8.5),
    ):
        selected = labels == label
        axes[1].scatter(
            points[selected, 0],
            points[selected, 1],
            s=size,
            color=colour,
            alpha=0.95,
            linewidths=0,
        )

    style_axis(axes[2])
    panel_segments = []
    for panel in lattice.panels:
        polygon = lattice_points[np.append(panel, panel[0])]
        panel_segments.extend(zip(polygon[:-1], polygon[1:]))
    axes[2].add_collection(
        LineCollection(panel_segments, colors=INK, linewidths=0.48, alpha=0.62)
    )
    trailing = lattice_points[lattice.wake_edges]
    wake_segments = []
    wake_extent = 1.15 * params.root_chord
    for endpoint in trailing.reshape(-1, 2):
        wake_segments.append([endpoint, endpoint + np.array([0.0, 0.62 * wake_extent])])
    axes[2].add_collection(
        LineCollection(wake_segments, colors=BLUE, linewidths=0.42, alpha=0.32)
    )
    axes[2].text(
        0.5,
        -0.06,
        r"$C_L$      $C_{Di}$      $C_M$",
        transform=axes[2].transAxes,
        color=INK,
        fontsize=14,
        ha="center",
        va="top",
    )

    x_limits = (points[:, 0].min() - 0.5, points[:, 0].max() + 0.5)
    y_limits = (points[:, 1].min() - 0.35, points[:, 1].max() + 0.9)
    for axis in axes:
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)

    figure.text(
        0.055,
        0.91,
        "FROM GEOMETRY TO AERODYNAMIC COEFFICIENTS",
        color=INK,
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.855,
        "Learning aerodynamic mesh semantics from an unstructured lifting surface",
        color="#6f6b64",
        fontsize=10,
    )
    for x, index, title in (
        (0.055, "01 / GEOMETRY", "Triangulated surface"),
        (0.375, "02 / SEMANTICS", "Aerodynamic meaning"),
        (0.695, "03 / COEFFICIENTS", "Vortex lattice and wake"),
    ):
        figure.text(x, 0.70, index, color=MUTED, fontsize=8, fontweight="bold")
        figure.text(x, 0.655, title, color=INK, fontsize=11, fontweight="bold")
    figure.text(0.342, 0.40, "→", color=MUTED, fontsize=20, ha="center")
    figure.text(0.662, 0.40, "→", color=MUTED, fontsize=20, ha="center")
    figure.subplots_adjust(left=0.05, right=0.97, top=0.59, bottom=0.10, wspace=0.14)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, facecolor=PAPER, bbox_inches="tight", pad_inches=0.22)
    plt.close(figure)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    generate()
