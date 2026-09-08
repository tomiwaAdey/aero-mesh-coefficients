from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

from .dataset import planform_envelope_edge_labels
from .geometry import LABEL_LEADING_EDGE, LABEL_TIP, LABEL_TRAILING_EDGE
from .stl import planform_coordinates
from .types import AeroMesh, TriMesh, WingParameters


@dataclass
class ReconstructionResult:
    mesh: AeroMesh | None
    geometry: WingParameters | None
    valid: bool
    leading_edge_points: int
    trailing_edge_points: int
    diagnostics: dict[str, float | int | str]
    failure_code: str | None = None
    tip_edge_points: int = 0
    boundary_paths: dict[str, np.ndarray] | None = None


def _failure(
    code: str,
    leading: int = 0,
    trailing: int = 0,
    tips: int = 0,
    **diagnostics: float | int | str,
) -> ReconstructionResult:
    return ReconstructionResult(
        mesh=None,
        geometry=None,
        valid=False,
        leading_edge_points=leading,
        trailing_edge_points=trailing,
        diagnostics={"reason_code": code, **diagnostics},
        failure_code=code,
        tip_edge_points=tips,
    )


def _polynomial_features(values: np.ndarray, degree: int = 3) -> np.ndarray:
    return np.column_stack([values**power for power in range(degree + 1)])


def _fit_curve(y: np.ndarray, values: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
    degree = min(5, max(1, len(np.unique(np.round(y, 8))) - 1))
    design = _polynomial_features(y, degree)
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    weights = np.ones(len(y), dtype=float)
    coefficients = np.zeros(design.shape[1], dtype=float)
    for _ in range(8):
        weighted = design * weights[:, None]
        coefficients = np.linalg.solve(weighted.T @ design + penalty, weighted.T @ values)
        residual = values - design @ coefficients
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9
        normalized = residual / (4.685 * scale)
        weights = np.where(np.abs(normalized) < 1.0, (1.0 - normalized**2) ** 2, 0.0)
        if np.sum(weights > 0.0) < design.shape[1] + 1:
            weights = np.ones(len(y), dtype=float)
            break
    return coefficients


def _fit_curve_without_outliers(y: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficients = _fit_curve(y, values)
    residual = values - _predict_curve(coefficients, y)
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9
    # The aerodynamic boundary can contain a real planform kink. Only reject
    # deviations too large to be explained by that resolved geometry.
    threshold = max(8.0 * scale, 0.20 * float(np.ptp(values)), 1e-5)
    keep = np.abs(residual - np.median(residual)) <= threshold
    if np.sum(keep) >= max(4, len(coefficients)):
        coefficients = _fit_curve(y[keep], values[keep])
    else:
        keep = np.ones(len(y), dtype=bool)
    return coefficients, keep


def _predict_curve(coefficients: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _polynomial_features(y, len(coefficients) - 1) @ coefficients


def _interpolate_boundary_curve(
    y: np.ndarray,
    values: np.ndarray,
    target_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter geometric outliers, then preserve the observed boundary shape."""
    coefficients, keep = _fit_curve_without_outliers(y, values)
    retained_y = np.asarray(y[keep], dtype=float)
    retained_values = np.asarray(values[keep], dtype=float)
    if len(retained_y) < 2:
        return _predict_curve(coefficients, target_y), keep
    order = np.argsort(retained_y)
    retained_y = retained_y[order]
    retained_values = retained_values[order]
    rounded = np.round(retained_y, 10)
    unique_y = np.unique(rounded)
    collapsed_values = np.asarray(
        [np.median(retained_values[rounded == value]) for value in unique_y], dtype=float
    )
    if len(unique_y) < 2:
        return _predict_curve(coefficients, target_y), keep
    return np.interp(target_y, unique_y, collapsed_values), keep


def _surface_features(
    y: np.ndarray,
    x_fraction: np.ndarray,
    y_center: float = 0.0,
    y_scale: float = 1.0,
) -> np.ndarray:
    normalized_y = (y - y_center) / max(y_scale, 1e-12)
    absolute_y = np.abs(normalized_y)
    return np.column_stack(
        [
            np.ones(len(y)),
            normalized_y,
            absolute_y,
            normalized_y**2,
            x_fraction,
            x_fraction**2,
            normalized_y * x_fraction,
            absolute_y * x_fraction,
            normalized_y**3,
            absolute_y**3,
            normalized_y**2 * x_fraction,
        ]
    )


def _probability_matrix(edge_labels: np.ndarray, probabilities: np.ndarray | None) -> np.ndarray:
    labels = np.asarray(edge_labels, dtype=int)
    if probabilities is None:
        result = np.full((len(labels), 4), 0.01, dtype=float)
        result[np.arange(len(labels)), labels] = 0.97
        return result / result.sum(axis=1, keepdims=True)
    result = np.asarray(probabilities, dtype=float)
    if result.shape != (len(labels), 4):
        raise ValueError("edge probabilities must have shape (edge_count, 4)")
    result = np.maximum(result, 1e-9)
    return result / result.sum(axis=1, keepdims=True)


def _shortest_probability_path(
    vertices: np.ndarray,
    edges: np.ndarray,
    scores: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    median_length = max(float(np.median(lengths)), 1e-12)
    adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(len(vertices))]
    for edge_index, ((left, right), length, score) in enumerate(zip(edges, lengths, scores)):
        cost = (length / median_length) * (0.05 - np.log(max(float(score), 1e-8)))
        adjacency[int(left)].append((int(right), edge_index, cost))
        adjacency[int(right)].append((int(left), edge_index, cost))
    distance = np.full(len(vertices), np.inf)
    previous_vertex = np.full(len(vertices), -1, dtype=int)
    previous_edge = np.full(len(vertices), -1, dtype=int)
    distance[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, int(start))]
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
        return np.empty(0, dtype=int), np.empty(0, dtype=int), 0.0
    path_vertices = [int(end)]
    path_edges = []
    current = int(end)
    while current != start:
        path_edges.append(int(previous_edge[current]))
        current = int(previous_vertex[current])
        if current < 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int), 0.0
        path_vertices.append(current)
    path_vertices.reverse()
    path_edges.reverse()
    confidence = float(np.mean(scores[path_edges])) if path_edges else 0.0
    return np.asarray(path_vertices), np.asarray(path_edges), confidence


def _span_boundary_path(
    surface: TriMesh,
    edges: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    midpoints = 0.5 * (surface.vertices[edges[:, 0]] + surface.vertices[edges[:, 1]])
    threshold = max(0.15, float(np.quantile(scores, 0.90)))
    candidates = np.flatnonzero(scores >= threshold)
    if len(candidates) < 2:
        candidates = np.argsort(scores)[-max(2, min(12, len(scores))):]
    left_edge = edges[candidates[np.argmin(midpoints[candidates, 1])]]
    right_edge = edges[candidates[np.argmax(midpoints[candidates, 1])]]
    start = int(left_edge[np.argmin(surface.vertices[left_edge, 1])])
    end = int(right_edge[np.argmax(surface.vertices[right_edge, 1])])
    return _shortest_probability_path(surface.vertices, edges, scores, start, end)


def recover_boundary_paths(
    surface: TriMesh,
    edges: np.ndarray,
    edge_labels: np.ndarray,
    edge_probabilities: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Recover connected leading, trailing and two tip paths from edge scores."""
    probabilities = _probability_matrix(edge_labels, edge_probabilities)
    midpoints = 0.5 * (surface.vertices[edges[:, 0]] + surface.vertices[edges[:, 1]])
    chord_fraction, span_fraction = planform_coordinates(surface.vertices, midpoints)
    leading_scores = probabilities[:, LABEL_LEADING_EDGE].copy()
    trailing_scores = probabilities[:, LABEL_TRAILING_EDGE].copy()
    tip_scores = probabilities[:, LABEL_TIP].copy()
    leading_scores[chord_fraction > 0.60] *= 1e-3
    trailing_scores[chord_fraction < 0.40] *= 1e-3
    tip_scores[span_fraction < 0.70] *= 1e-3

    leading_vertices, leading_edges, leading_confidence = _span_boundary_path(
        surface, edges, leading_scores
    )
    trailing_vertices, trailing_edges, trailing_confidence = _span_boundary_path(
        surface, edges, trailing_scores
    )
    paths: dict[str, np.ndarray] = {
        "leading": leading_vertices,
        "trailing": trailing_vertices,
    }
    diagnostics = {
        "leading_path_confidence": leading_confidence,
        "trailing_path_confidence": trailing_confidence,
        "leading_path_edges": float(len(leading_edges)),
        "trailing_path_edges": float(len(trailing_edges)),
    }
    if not len(leading_vertices) or not len(trailing_vertices):
        return paths, diagnostics

    for side, name in (("left", "tip_left"), ("right", "tip_right")):
        if side == "left":
            start = int(leading_vertices[np.argmin(surface.vertices[leading_vertices, 1])])
            end = int(trailing_vertices[np.argmin(surface.vertices[trailing_vertices, 1])])
        else:
            start = int(leading_vertices[np.argmax(surface.vertices[leading_vertices, 1])])
            end = int(trailing_vertices[np.argmax(surface.vertices[trailing_vertices, 1])])
        vertices, path_edges, confidence = _shortest_probability_path(
            surface.vertices, edges, tip_scores, start, end
        )
        paths[name] = vertices
        diagnostics[f"{name}_confidence"] = confidence
        diagnostics[f"{name}_edges"] = float(len(path_edges))
    return paths, diagnostics


def _paired_camber_observations(
    surface: TriMesh,
    fraction: np.ndarray,
    y_center: float,
    y_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = surface.vertices
    planar = np.column_stack([(vertices[:, 1] - y_center) / max(y_scale, 1e-12), fraction])
    tree = cKDTree(planar)
    distances, neighbours = tree.query(planar, k=min(24, len(vertices)))
    triangles = vertices[surface.faces]
    weighted_face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    vertex_normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(vertex_normals, surface.faces[:, corner], weighted_face_normals)
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1)[:, None], 1e-12)
    used: set[tuple[int, int]] = set()
    rows: list[tuple[float, float, float]] = []
    for index in range(len(vertices)):
        if abs(vertex_normals[index, 2]) < 0.20:
            continue
        candidates = np.atleast_1d(neighbours[index])[1:]
        candidate_distances = np.atleast_1d(distances[index])[1:]
        opposite = vertex_normals[candidates, 2] * vertex_normals[index, 2] < 0.0
        well_oriented = np.abs(vertex_normals[candidates, 2]) >= 0.20
        close_mask = (candidate_distances <= 0.05) & opposite & well_oriented
        close = candidates[close_mask]
        if not len(close):
            continue
        close_distances = candidate_distances[close_mask]
        partner = int(close[np.argmin(close_distances)])
        pair = tuple(sorted((index, partner)))
        if pair in used or abs(vertices[index, 2] - vertices[partner, 2]) < 1e-7:
            continue
        used.add(pair)
        rows.append(
            (
                float(0.5 * (vertices[index, 1] + vertices[partner, 1])),
                float(0.5 * (fraction[index] + fraction[partner])),
                float(0.5 * (vertices[index, 2] + vertices[partner, 2])),
            )
        )
    if not rows:
        return np.asarray([]), np.asarray([]), np.asarray([])
    values = np.asarray(rows, dtype=float)
    return values[:, 0], values[:, 1], values[:, 2]


def _interpolate_camber_surface(
    y: np.ndarray,
    fraction: np.ndarray,
    camber: np.ndarray,
    y_grid: np.ndarray,
    fraction_grid: np.ndarray,
    y_center: float,
    y_scale: float,
) -> tuple[np.ndarray, float]:
    """Interpolate the paired upper/lower mean surface onto the aerodynamic grid."""
    points = np.column_stack([(y - y_center) / max(y_scale, 1e-12), fraction])
    rounded = np.round(points, 10)
    unique_points, inverse = np.unique(rounded, axis=0, return_inverse=True)
    unique_camber = np.asarray(
        [np.mean(camber[inverse == index]) for index in range(len(unique_points))], dtype=float
    )
    if len(unique_points) < 3 or np.linalg.matrix_rank(unique_points - unique_points.mean(axis=0)) < 2:
        raise ValueError("camber observations do not span the lifting surface")
    linear = LinearNDInterpolator(unique_points, unique_camber, fill_value=np.nan)
    nearest = NearestNDInterpolator(unique_points, unique_camber)
    grid_y, grid_fraction = np.meshgrid(
        (y_grid - y_center) / max(y_scale, 1e-12), fraction_grid, indexing="ij"
    )
    targets = np.column_stack([grid_y.ravel(), grid_fraction.ravel()])
    values = np.asarray(linear(targets), dtype=float)
    missing = ~np.isfinite(values)
    if np.any(missing):
        values[missing] = np.asarray(nearest(targets[missing]), dtype=float)
    fitted = np.asarray(linear(unique_points), dtype=float)
    missing_fitted = ~np.isfinite(fitted)
    if np.any(missing_fitted):
        fitted[missing_fitted] = np.asarray(nearest(unique_points[missing_fitted]), dtype=float)
    fit_rmse = float(np.sqrt(np.mean((fitted - unique_camber) ** 2)))
    return values.reshape((len(y_grid), len(fraction_grid))), fit_rmse


def validate_aero_mesh(mesh: AeroMesh) -> tuple[bool, str | None, dict[str, float]]:
    panels = mesh.vertices[mesh.panels]
    first = 0.5 * np.linalg.norm(
        np.cross(panels[:, 1] - panels[:, 0], panels[:, 3] - panels[:, 0]), axis=1
    )
    second = 0.5 * np.linalg.norm(
        np.cross(panels[:, 2] - panels[:, 1], panels[:, 3] - panels[:, 1]), axis=1
    )
    areas = first + second
    if np.any(areas <= 1e-12) or np.any(mesh.normals[:, 2] <= 0.0):
        return False, "INVALID_PANEL_ORIENTATION", {"minimum_panel_area": float(np.min(areas))}

    convex = np.ones(len(panels), dtype=bool)
    for index, (panel, normal) in enumerate(zip(panels, mesh.normals)):
        edge_vectors = np.roll(panel, -1, axis=0) - panel
        turns = np.array(
            [np.dot(np.cross(edge_vectors[j], edge_vectors[(j + 1) % 4]), normal) for j in range(4)]
        )
        convex[index] = np.all(turns >= -1e-10) or np.all(turns <= 1e-10)
    if not np.all(convex):
        return False, "INVALID_PANEL_CONVEXITY", {"convex_panel_fraction": float(np.mean(convex))}

    edge_counts: dict[tuple[int, int], int] = {}
    for panel in mesh.panels:
        for start, end in ((panel[0], panel[1]), (panel[1], panel[2]), (panel[2], panel[3]), (panel[3], panel[0])):
            edge = tuple(sorted((int(start), int(end))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    if any(count > 2 for count in edge_counts.values()):
        return False, "INVALID_NEIGHBOURS", {"maximum_panel_edge_uses": float(max(edge_counts.values()))}
    expected_boundary = 2 * int(mesh.metadata["n_span"]) + 2 * int(mesh.metadata["n_chord"])
    actual_boundary = sum(count == 1 for count in edge_counts.values())
    if actual_boundary != expected_boundary:
        return False, "INVALID_NEIGHBOURS", {"boundary_panel_edges": float(actual_boundary)}
    return True, None, {
        "minimum_panel_area": float(np.min(areas)),
        "convex_panel_fraction": float(np.mean(convex)),
        "boundary_panel_edges": float(actual_boundary),
    }


def _grid_to_aero_mesh(grid: np.ndarray) -> AeroMesh:
    n_span = grid.shape[0] - 1
    n_chord = grid.shape[1] - 1
    vertices = grid.reshape((-1, 3))
    panels = []
    collocation = []
    normals = []
    bound_left = []
    bound_right = []
    span_widths = []
    wake_edges = []

    def index(span_index: int, chord_index: int) -> int:
        return span_index * (n_chord + 1) + chord_index

    for i in range(n_span):
        for j in range(n_chord):
            panel = [index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)]
            panels.append(panel)
            p00, p10, p11, p01 = vertices[panel]
            chord_left = p01 - p00
            chord_right = p11 - p10
            span_front = p10 - p00
            normal = np.cross(chord_left + chord_right, span_front)
            normal /= max(np.linalg.norm(normal), 1e-12)
            if normal[2] < 0:
                normal = -normal
            normals.append(normal)
            collocation.append(0.125 * (p00 + p10) + 0.375 * (p01 + p11))
            bound_left.append(0.75 * p00 + 0.25 * p01)
            bound_right.append(0.75 * p10 + 0.25 * p11)
            span_widths.append(float(np.linalg.norm(p10 - p00)))
            if j == n_chord - 1:
                wake_edges.append([len(panels) - 1, panel[1], panel[2]])

    panel_array = np.asarray(panels, dtype=int)
    surface_area = 0.0
    for panel in panel_array:
        points = vertices[panel]
        surface_area += 0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[3] - points[0]))
        surface_area += 0.5 * np.linalg.norm(np.cross(points[2] - points[1], points[3] - points[1]))
    y = grid[:, 0, 1]
    chord = grid[:, -1, 0] - grid[:, 0, 0]
    reference_area = float(np.trapezoid(chord, y))
    reference_chord = float(np.trapezoid(chord**2, y) / max(reference_area, 1e-12))
    quarter_chord = grid[:, 0, 0] + 0.25 * chord
    reference_x = float(np.trapezoid(quarter_chord * chord, y) / max(reference_area, 1e-12))
    return AeroMesh(
        vertices=vertices,
        panels=panel_array,
        collocation_points=np.asarray(collocation),
        normals=np.asarray(normals),
        bound_left=np.asarray(bound_left),
        bound_right=np.asarray(bound_right),
        span_widths=np.asarray(span_widths),
        reference_area=reference_area,
        reference_chord=reference_chord,
        reference_span=float(np.ptp(y)),
        reference_point=np.array([reference_x, 0.0, 0.0], dtype=float),
        wake_edges=np.asarray(wake_edges, dtype=int),
        body_axes=np.eye(3),
        metadata={
            "reconstructed": True,
            "surface_area": surface_area,
            "n_span": n_span,
            "n_chord": n_chord,
        },
    )


def reconstruct_aero_mesh_from_edges(
    surface: TriMesh,
    edges: np.ndarray,
    edge_labels: np.ndarray,
    n_span: int = 20,
    n_chord: int = 4,
    edge_probabilities: np.ndarray | None = None,
) -> ReconstructionResult:
    """Recover a structured lifting-surface mesh from classified STL edges."""
    paths, path_diagnostics = recover_boundary_paths(
        surface, edges, edge_labels, edge_probabilities=edge_probabilities
    )
    leading_vertices = paths.get("leading", np.empty(0, dtype=int))
    trailing_vertices = paths.get("trailing", np.empty(0, dtype=int))
    tip_left = paths.get("tip_left", np.empty(0, dtype=int))
    tip_right = paths.get("tip_right", np.empty(0, dtype=int))
    tip_count = int(len(tip_left) + len(tip_right))
    minimum_points = max(6, n_span // 3)
    if len(leading_vertices) < minimum_points or len(trailing_vertices) < minimum_points:
        return _failure(
            "MISSING_BOUNDARY_PATH", len(leading_vertices), len(trailing_vertices), tip_count, **path_diagnostics
        )
    if min(path_diagnostics["leading_path_confidence"], path_diagnostics["trailing_path_confidence"]) < 0.15:
        return _failure(
            "LOW_BOUNDARY_CONFIDENCE", len(leading_vertices), len(trailing_vertices), tip_count, **path_diagnostics
        )
    if len(tip_left) < 3 or len(tip_right) < 3:
        return _failure(
            "MISSING_TIP_PATH", len(leading_vertices), len(trailing_vertices), tip_count, **path_diagnostics
        )
    if min(path_diagnostics["tip_left_confidence"], path_diagnostics["tip_right_confidence"]) < 0.12:
        return _failure(
            "LOW_TIP_CONFIDENCE", len(leading_vertices), len(trailing_vertices), tip_count, **path_diagnostics
        )

    le_points = surface.vertices[leading_vertices]
    te_points = surface.vertices[trailing_vertices]
    left_points = surface.vertices[tip_left]
    right_points = surface.vertices[tip_right]
    span_min = float(np.median(left_points[:, 1]))
    span_max = float(np.median(right_points[:, 1]))
    if span_min > span_max:
        span_min, span_max = span_max, span_min
    if span_max - span_min < 1e-6:
        return _failure("INVALID_SPAN", len(leading_vertices), len(trailing_vertices), tip_count)
    span_extent = span_max - span_min

    # The initial pose estimate deliberately precedes semantic recognition. Once
    # both aerodynamic boundaries are known, remove its remaining section-plane
    # ambiguity by aligning the recovered root chord with the body x axis.
    root_y = 0.5 * (span_min + span_max)
    le_root_x, _ = _interpolate_boundary_curve(
        le_points[:, 1], le_points[:, 0], np.asarray([root_y])
    )
    le_root_z, _ = _interpolate_boundary_curve(
        le_points[:, 1], le_points[:, 2], np.asarray([root_y])
    )
    te_root_x, _ = _interpolate_boundary_curve(
        te_points[:, 1], te_points[:, 0], np.asarray([root_y])
    )
    te_root_z, _ = _interpolate_boundary_curve(
        te_points[:, 1], te_points[:, 2], np.asarray([root_y])
    )
    root_chord_vector = np.array(
        [te_root_x[0] - le_root_x[0], 0.0, te_root_z[0] - le_root_z[0]], dtype=float
    )
    if np.linalg.norm(root_chord_vector) <= 1e-8:
        return _failure("INVALID_ROOT_CHORD", len(leading_vertices), len(trailing_vertices), tip_count)
    chord_axis = root_chord_vector / np.linalg.norm(root_chord_vector)
    span_axis = np.array([0.0, 1.0, 0.0])
    thickness_axis = np.cross(chord_axis, span_axis)
    thickness_axis /= max(float(np.linalg.norm(thickness_axis)), 1e-12)
    refined_axes = np.column_stack([chord_axis, span_axis, thickness_axis])
    chord_frame_rotation_deg = float(
        np.degrees(np.arctan2(root_chord_vector[2], root_chord_vector[0]))
    )
    surface = TriMesh(
        surface.vertices @ refined_axes,
        surface.faces,
        surface.labels,
        surface.metadata,
    )
    le_points = surface.vertices[leading_vertices]
    te_points = surface.vertices[trailing_vertices]
    left_points = surface.vertices[tip_left]
    right_points = surface.vertices[tip_right]
    span_min = float(np.median(left_points[:, 1]))
    span_max = float(np.median(right_points[:, 1]))
    if span_min > span_max:
        span_min, span_max = span_max, span_min
    span_extent = span_max - span_min
    le_coverage = float(np.ptp(le_points[:, 1]) / span_extent)
    te_coverage = float(np.ptp(te_points[:, 1]) / span_extent)
    if min(le_coverage, te_coverage) < 0.75:
        return _failure(
            "LOW_BOUNDARY_COVERAGE",
            len(leading_vertices),
            len(trailing_vertices),
            tip_count,
            leading_coverage=le_coverage,
            trailing_coverage=te_coverage,
        )

    all_y = surface.vertices[:, 1]
    all_le, le_x_keep = _interpolate_boundary_curve(
        le_points[:, 1], le_points[:, 0], all_y
    )
    all_te, te_x_keep = _interpolate_boundary_curve(
        te_points[:, 1], te_points[:, 0], all_y
    )
    chord = all_te - all_le
    if np.mean(chord > 1e-4) < 0.8:
        return _failure("NONPOSITIVE_CHORD", len(leading_vertices), len(trailing_vertices), tip_count)
    all_fraction = np.clip((surface.vertices[:, 0] - all_le) / np.maximum(chord, 1e-6), 0.0, 1.0)
    y_center = 0.5 * (span_min + span_max)
    y_scale = 0.5 * span_extent
    paired_y, paired_fraction, paired_camber = _paired_camber_observations(
        surface, all_fraction, y_center, y_scale
    )
    if len(paired_y) < 20:
        return _failure(
            "INSUFFICIENT_CAMBER_PAIRS",
            len(leading_vertices),
            len(trailing_vertices),
            tip_count,
            paired_camber_points=len(paired_y),
        )
    paired_le_z, _ = _interpolate_boundary_curve(
        le_points[:, 1], le_points[:, 2], paired_y
    )
    paired_te_z, _ = _interpolate_boundary_curve(
        te_points[:, 1], te_points[:, 2], paired_y
    )
    camber_from_chord = paired_camber - (
        paired_le_z + paired_fraction * (paired_te_z - paired_le_z)
    )
    interior_camber = (paired_fraction >= 0.15) & (paired_fraction <= 0.85)
    camber_sign_metric = float(
        np.median(camber_from_chord[interior_camber])
        if np.any(interior_camber)
        else np.median(camber_from_chord)
    )
    thickness_frame_flipped = camber_sign_metric < -1e-5
    if thickness_frame_flipped:
        vertices = surface.vertices.copy()
        vertices[:, 2] *= -1.0
        surface = TriMesh(vertices, surface.faces, surface.labels, surface.metadata)
        le_points = surface.vertices[leading_vertices]
        te_points = surface.vertices[trailing_vertices]
        paired_camber *= -1.0
        camber_from_chord *= -1.0
        camber_sign_metric *= -1.0
    y_grid = np.linspace(span_min, span_max, n_span + 1)
    chord_grid = np.linspace(0.0, 1.0, n_chord + 1)
    grid = np.zeros((n_span + 1, n_chord + 1, 3), dtype=float)
    le_x, _ = _interpolate_boundary_curve(le_points[:, 1], le_points[:, 0], y_grid)
    te_x, _ = _interpolate_boundary_curve(te_points[:, 1], te_points[:, 0], y_grid)
    le_z, le_z_keep = _interpolate_boundary_curve(
        le_points[:, 1], le_points[:, 2], y_grid
    )
    te_z, te_z_keep = _interpolate_boundary_curve(
        te_points[:, 1], te_points[:, 2], y_grid
    )
    reconstructed_chord = te_x - le_x
    observed_chord_extent = max(float(np.ptp(surface.vertices[:, 0])), 1e-12)
    if np.min(reconstructed_chord) <= 1e-4:
        return _failure(
            "NONPOSITIVE_CHORD",
            len(leading_vertices),
            len(trailing_vertices),
            tip_count,
            minimum_reconstructed_chord=float(np.min(reconstructed_chord)),
        )
    if np.max(reconstructed_chord) > 1.5 * observed_chord_extent:
        return _failure(
            "IMPLAUSIBLE_CHORD",
            len(leading_vertices),
            len(trailing_vertices),
            tip_count,
            maximum_reconstructed_chord=float(np.max(reconstructed_chord)),
        )
    leading_fraction, _ = planform_coordinates(surface.vertices, le_points)
    trailing_fraction, _ = planform_coordinates(surface.vertices, te_points)
    median_leading_fraction = float(np.median(leading_fraction))
    median_trailing_fraction = float(np.median(trailing_fraction))
    if median_leading_fraction > 0.15 or median_trailing_fraction < 0.85:
        return _failure(
            "IMPLAUSIBLE_CHORD",
            len(leading_vertices),
            len(trailing_vertices),
            tip_count,
            minimum_reconstructed_chord=float(np.min(reconstructed_chord)),
            maximum_reconstructed_chord=float(np.max(reconstructed_chord)),
            median_leading_planform_fraction=median_leading_fraction,
            median_trailing_planform_fraction=median_trailing_fraction,
        )
    try:
        camber_grid, surface_fit_rmse = _interpolate_camber_surface(
            paired_y,
            paired_fraction,
            camber_from_chord,
            y_grid,
            chord_grid,
            y_center,
            y_scale,
        )
    except ValueError:
        return _failure(
            "INSUFFICIENT_CAMBER_COVERAGE",
            len(leading_vertices),
            len(trailing_vertices),
            tip_count,
            paired_camber_points=len(paired_y),
        )
    for i, y_value in enumerate(y_grid):
        fractions = chord_grid
        x_values = le_x[i] + fractions * reconstructed_chord[i]
        z_values = (
            le_z[i]
            + fractions * (te_z[i] - le_z[i])
            + camber_grid[i]
        )
        z_values[0] = le_z[i]
        z_values[-1] = te_z[i]
        grid[i, :, 0] = x_values
        grid[i, :, 1] = y_value
        grid[i, :, 2] = z_values

    mesh = _grid_to_aero_mesh(grid)
    mesh_valid, mesh_failure, mesh_diagnostics = validate_aero_mesh(mesh)
    if not mesh_valid:
        return _failure(
            mesh_failure or "INVALID_AERO_MESH",
            len(leading_vertices),
            len(trailing_vertices),
            tip_count,
            **mesh_diagnostics,
        )
    root_index = int(np.argmin(np.abs(y_grid)))
    root_chord = float(reconstructed_chord[root_index])
    tip_chord = float(0.5 * (reconstructed_chord[0] + reconstructed_chord[-1]))
    geometry = WingParameters(
        name="reconstructed_stl",
        span=span_extent,
        root_chord=max(root_chord, 1e-4),
        tip_chord=max(tip_chord, 1e-4),
        n_span=n_span,
        n_chord=n_chord,
    )
    diagnostics: dict[str, float | int | str] = {
        "minimum_reconstructed_chord": float(np.min(reconstructed_chord)),
        "maximum_reconstructed_chord": float(np.max(reconstructed_chord)),
        "median_leading_planform_fraction": median_leading_fraction,
        "median_trailing_planform_fraction": median_trailing_fraction,
        "surface_fit_rmse": surface_fit_rmse,
        "leading_coverage": le_coverage,
        "trailing_coverage": te_coverage,
        "leading_inlier_fraction": float(np.mean(le_x_keep & le_z_keep)),
        "trailing_inlier_fraction": float(np.mean(te_x_keep & te_z_keep)),
        "paired_camber_points": int(len(paired_y)),
        "chord_frame_rotation_deg": chord_frame_rotation_deg,
        "camber_sign_metric": camber_sign_metric,
        "thickness_frame_flipped": int(thickness_frame_flipped),
        **path_diagnostics,
        **mesh_diagnostics,
    }
    return ReconstructionResult(
        mesh=mesh,
        geometry=geometry,
        valid=True,
        leading_edge_points=int(len(leading_vertices)),
        trailing_edge_points=int(len(trailing_vertices)),
        diagnostics=diagnostics,
        tip_edge_points=tip_count,
        boundary_paths=paths,
    )


def reconstruct_from_adaptive_planform_envelope(
    surface: TriMesh,
    edges: np.ndarray,
    features: np.ndarray,
    margins: tuple[float, ...] = (0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100),
) -> tuple[ReconstructionResult, np.ndarray, float]:
    """Use the narrowest local envelope that yields a valid aerodynamic mesh."""
    last_result = _failure("NO_VALID_ENVELOPE")
    last_labels = np.zeros(len(edges), dtype=int)
    for margin in margins:
        labels = planform_envelope_edge_labels(features, margin=margin)
        result = reconstruct_aero_mesh_from_edges(surface, edges, labels)
        if result.valid:
            result.diagnostics["envelope_margin"] = margin
            return result, labels, margin
        last_result = result
        last_labels = labels
    last_result.diagnostics["envelope_margin"] = margins[-1]
    return last_result, last_labels, margins[-1]


def source_boundary_errors(
    surface: TriMesh,
    result: ReconstructionResult,
) -> dict[str, float]:
    """Measure recovered paths against immutable source curves in root-chord units."""
    if result.boundary_paths is None or "source_curves" not in surface.metadata:
        return {}
    curves = surface.metadata["source_curves"]
    root_chord = max(float(surface.metadata.get("source_root_chord", 1.0)), 1e-12)
    leading = surface.vertices[result.boundary_paths["leading"]]
    trailing = surface.vertices[result.boundary_paths["trailing"]]
    tip_left = surface.vertices[result.boundary_paths["tip_left"]]
    tip_right = surface.vertices[result.boundary_paths["tip_right"]]
    reference_tips = sorted(curves["tips"], key=lambda values: float(np.mean(values[:, 1])))
    estimated_tips = sorted((tip_left, tip_right), key=lambda values: float(np.mean(values[:, 1])))
    option_groups = surface.metadata.get("source_tip_boundary_options")
    if option_groups is not None:
        option_groups = sorted(
            option_groups,
            key=lambda options: float(np.mean(np.asarray(options[0])[:, 1])),
        )
        tip_errors = [
            min(
                symmetric_curve_distance(np.asarray(option), estimate)
                for option in options
            )
            / root_chord
            for options, estimate in zip(option_groups, estimated_tips)
        ]
    else:
        tip_errors = [
            symmetric_curve_distance(np.asarray(reference), estimate) / root_chord
            for reference, estimate in zip(reference_tips, estimated_tips)
        ]
    return {
        "leading_boundary_error_root_chords": symmetric_curve_distance(
            np.asarray(curves["leading_edge"]), leading
        )
        / root_chord,
        "trailing_boundary_error_root_chords": symmetric_curve_distance(
            np.asarray(curves["trailing_edge"]), trailing
        )
        / root_chord,
        "tip_boundary_error_root_chords": float(np.mean(tip_errors)),
    }


def symmetric_curve_distance(reference: np.ndarray, estimate: np.ndarray) -> float:
    if not len(reference) or not len(estimate):
        return float("inf")
    return float(
        0.5
        * (
            np.mean(_point_to_polyline_distance(reference, estimate))
            + np.mean(_point_to_polyline_distance(estimate, reference))
        )
    )


def _point_to_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Return each point's shortest Euclidean distance to a piecewise-linear curve."""
    points = np.asarray(points, dtype=float)
    polyline = np.asarray(polyline, dtype=float)
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    offsets = points[:, None, :] - starts[None, :, :]
    parameters = np.divide(
        np.einsum("pij,ij->pi", offsets, vectors),
        lengths_squared[None, :],
        out=np.zeros((len(points), len(vectors)), dtype=float),
        where=lengths_squared[None, :] > 1e-24,
    )
    parameters = np.clip(parameters, 0.0, 1.0)
    projections = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - projections, axis=2), axis=1)
