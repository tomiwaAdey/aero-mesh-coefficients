from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from aero_mesh.classical_models import predict_class_probabilities
from aero_mesh.dataset import generate_geometry_cases, split_geometry_cases
from aero_mesh.reconstruction import reconstruct_aero_mesh_from_edges

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"


def generate_method_pipeline() -> None:
    """Draw the paper's method flowchart in the visual style of the source report."""
    OUT.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.2, 11.5), dpi=180)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 16)
    axis.axis("off")
    blue = "#4f80d1"

    steps = [
        (14.7, "Read triangulated surface", 4.0),
        (13.0, "Establish canonical frame of reference", 5.5),
        (11.3, "Calculate 19 quantities for every edge", 5.7),
        (9.6, "Classify leading edge, trailing edge and tips", 6.2),
        (7.9, "Recover connected paths and interpolate boundaries", 6.4),
        (6.2, "Pair surfaces, generate and check aerodynamic mesh", 6.4),
        (4.0, "Populate influence matrix and solve circulation", 6.2),
        (2.3, "Calculate aerodynamic coefficients", 5.2),
        (0.8, "Output", 2.1),
    ]
    boxes: list[tuple[float, float, float, float]] = []
    for y_position, label, width in steps:
        height = 0.72
        x_position = 4.65 - width / 2
        axis.add_patch(
            Rectangle(
                (x_position, y_position),
                width,
                height,
                facecolor="white",
                edgecolor=blue,
                linewidth=0.8,
            )
        )
        axis.text(
            4.65,
            y_position + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            family="sans-serif",
        )
        boxes.append((x_position, y_position, width, height))

    for upper, lower in zip(boxes[:-1], boxes[1:]):
        axis.add_patch(
            FancyArrowPatch(
                (4.65, upper[1]),
                (4.65, lower[1] + lower[3]),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=0.75,
                color="black",
            )
        )

    validity = boxes[5]
    fallback_x, fallback_y, fallback_width, fallback_height = 8.0, 6.15, 1.75, 0.82
    axis.add_patch(
        Rectangle(
            (fallback_x, fallback_y),
            fallback_width,
            fallback_height,
            facecolor="white",
            edgecolor=blue,
            linewidth=0.8,
        )
    )
    axis.text(
        fallback_x + fallback_width / 2,
        fallback_y + fallback_height / 2,
        "Geometrical\nfallback",
        ha="center",
        va="center",
        fontsize=9.5,
    )
    axis.add_patch(
        FancyArrowPatch(
            (validity[0] + validity[2], validity[1] + validity[3] / 2),
            (fallback_x, fallback_y + fallback_height / 2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.75,
            color="black",
        )
    )
    axis.text(
        7.72,
        6.67,
        "invalid",
        ha="center",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},
    )
    axis.plot(
        [fallback_x + fallback_width / 2, fallback_x + fallback_width / 2, 5.75],
        [fallback_y, 5.35, 5.35],
        color="black",
        linewidth=0.75,
    )
    axis.add_patch(
        FancyArrowPatch(
            (5.75, 5.35),
            (5.75, validity[1]),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.75,
            color="black",
        )
    )
    axis.text(4.95, 5.7, "valid", ha="left", va="center", fontsize=8.5)

    figure.savefig(OUT / "method_pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def generate_geometry_split() -> None:
    """Show the complete-geometry train, validation and test split."""
    OUT.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11.5, 6.4), dpi=180)
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7)
    axis.axis("off")

    blue = "#4f80d1"
    blue_fill = "#eaf1fc"
    grey = "#5b5b5b"

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        detail: str,
        *,
        highlighted: bool = False,
    ) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                facecolor=blue_fill if highlighted else "white",
                edgecolor=blue if highlighted else grey,
                linewidth=0.9,
            )
        )
        axis.text(
            x + width / 2,
            y + height * 0.63,
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            family="monospace",
            weight="bold",
        )
        axis.text(
            x + width / 2,
            y + height * 0.31,
            detail,
            ha="center",
            va="center",
            fontsize=9.5,
            family="monospace",
            color="#303030",
        )

    axis.add_patch(
        FancyBboxPatch(
            (4.1, 6.2),
            3.8,
            0.48,
            boxstyle="round,pad=0.02,rounding_size=0.24",
            facecolor=blue_fill,
            edgecolor=blue,
            linewidth=0.9,
        )
    )
    axis.text(
        6.0,
        6.44,
        "REPEATED FOR SEEDS 13, 29, 47, 61 AND 79",
        ha="center",
        va="center",
        fontsize=9.6,
        family="monospace",
        weight="bold",
    )

    box(0.7, 4.55, 4.1, 1.05, "FAMILIAR PLANFORMS", "55 trapezoidal + 41 elliptical")
    box(7.2, 4.55, 4.1, 1.05, "UNSEEN PLANFORM FAMILY", "20 cranked")

    lower_boxes = [
        (0.05, "TRAIN", "60 wings x 3 meshes", False),
        (2.43, "VALIDATE", "18 wings x 3 meshes", False),
        (4.81, "IID TEST", "18 wings x 3 meshes", True),
        (7.19, "UNSEEN MESHER", "18 wings x 1 mesh", True),
        (9.57, "FAMILY SHIFT", "20 wings x 3 meshes", True),
    ]
    for x, title, detail, highlighted in lower_boxes:
        box(x, 1.75, 2.30, 1.05, title, detail, highlighted=highlighted)

    for target_x in (1.20, 3.58, 5.96, 8.34):
        axis.add_patch(
            FancyArrowPatch(
                (2.75, 4.55),
                (target_x, 2.8),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.8,
                color=grey,
                connectionstyle="arc3,rad=0.0",
            )
        )
    axis.add_patch(
        FancyArrowPatch(
            (9.25, 4.55),
            (10.72, 2.8),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.8,
            color=grey,
        )
    )

    axis.plot([0.65, 11.35], [0.92, 0.92], color="#b8b8b8", linewidth=0.7)
    axis.text(
        6.0,
        0.55,
        "ALL REMESHINGS OF A PHYSICAL WING STAY TOGETHER · NO WING CROSSES A SPLIT",
        ha="center",
        va="center",
        fontsize=9.4,
        family="monospace",
        color="#3c3c3c",
    )

    figure.savefig(OUT / "geometry_split.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _geometry_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _draw_surface(axis: plt.Axes, vertices: np.ndarray, faces: np.ndarray) -> None:
    axis.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color="#edf1f6",
        edgecolor="#747b84",
        linewidth=0.18,
        alpha=0.92,
        shade=False,
    )


def _draw_lattice(axis: plt.Axes, vertices: np.ndarray, panels: np.ndarray) -> None:
    for panel in panels:
        points = vertices[np.r_[panel, panel[0]]]
        axis.plot(points[:, 0], points[:, 1], points[:, 2], color="#3567b7", linewidth=0.65)


def _finish_geometry_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=10.5, pad=7, loc="left")
    axis.set_axis_off()
    axis.view_init(elev=24, azim=-62)
    axis.set_box_aspect((1.5, 3.0, 0.45))


def generate_geometry_sequence() -> None:
    """Generate the input-to-lattice sequence from the executable pipeline."""
    OUT.mkdir(parents=True, exist_ok=True)
    cases = generate_geometry_cases(
        {"trapezoidal": 55, "elliptical": 41, "cranked": 20}, seed=13
    )
    splits = split_geometry_cases(cases)

    metadata_path = ROOT / "results" / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_name = metadata["selected_feature_models"]["13"]
    model_path = ROOT / "results" / "models" / "selected-seed-13.joblib"
    classifier = joblib.load(model_path)
    case = splits["family_shift"][0]
    predictions = np.asarray(classifier.predict(case.features), dtype=int)
    probabilities = predict_class_probabilities(classifier, case.features)
    reconstruction = reconstruct_aero_mesh_from_edges(
        case.mesh,
        case.edges,
        predictions,
        edge_probabilities=probabilities,
    )
    if not reconstruction.valid or reconstruction.mesh is None:
        raise RuntimeError("The fixed Figure 2 geometry did not produce a valid lattice")

    edge_midpoints = 0.5 * (
        case.mesh.vertices[case.edges[:, 0]] + case.mesh.vertices[case.edges[:, 1]]
    )
    boundary = predictions > 0
    label_colours = {1: "#3567b7", 2: "#cf4d4d", 3: "#111111"}

    figure = plt.figure(figsize=(11.5, 7.0), dpi=180)
    axes = [figure.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]

    _draw_surface(axes[0], case.mesh.vertices, case.mesh.faces)
    _finish_geometry_axis(axes[0], "(a) Unstructured triangular input")

    axes[1].scatter(
        edge_midpoints[:, 0],
        edge_midpoints[:, 1],
        edge_midpoints[:, 2],
        c=case.features[:, 7],
        cmap="Greys",
        s=1.5,
        alpha=0.75,
    )
    _finish_geometry_axis(axes[1], "(b) Local edge descriptors")

    axes[2].scatter(
        edge_midpoints[~boundary, 0],
        edge_midpoints[~boundary, 1],
        edge_midpoints[~boundary, 2],
        color="#d8d8d8",
        s=0.6,
        alpha=0.18,
    )
    for label, colour in label_colours.items():
        selected = predictions == label
        axes[2].scatter(
            edge_midpoints[selected, 0],
            edge_midpoints[selected, 1],
            edge_midpoints[selected, 2],
            color=colour,
            s=5.0,
            alpha=0.9,
        )
    _finish_geometry_axis(axes[2], "(c) Classified aerodynamic boundaries")

    _draw_lattice(axes[3], reconstruction.mesh.vertices, reconstruction.mesh.panels)
    _finish_geometry_axis(axes[3], "(d) Reconstructed aerodynamic lattice")

    figure.subplots_adjust(
        left=0.01, right=0.99, top=0.96, bottom=0.02, wspace=0.02, hspace=0.10
    )
    output = OUT / "geometry_reconstruction_sequence.png"
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure_metadata = {
        "case": case.name,
        "family": case.family,
        "dataset_seed": 13,
        "classifier": model_name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "geometry_sha256": _geometry_hash(case.mesh.vertices, case.mesh.faces),
        "prediction_sha256": hashlib.sha256(
            np.ascontiguousarray(predictions, dtype=np.int64).tobytes()
        ).hexdigest(),
        "lattice_sha256": _geometry_hash(
            reconstruction.mesh.vertices, reconstruction.mesh.panels
        ),
        "figure_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (OUT / "geometry_reconstruction_sequence.json").write_text(
        json.dumps(figure_metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    generate_method_pipeline()
    generate_geometry_split()
    generate_geometry_sequence()
    print(f"Updated {OUT / 'method_pipeline.png'}")
    print(f"Updated {OUT / 'geometry_split.png'}")
    print(f"Updated {OUT / 'geometry_reconstruction_sequence.png'}")


if __name__ == "__main__":
    main()
