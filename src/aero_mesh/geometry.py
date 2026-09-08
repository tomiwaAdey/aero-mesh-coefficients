from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy.spatial import Delaunay

from .types import AeroMesh, GeometryFeatures, TriMesh, WingParameters

LABEL_SURFACE = 0
LABEL_LEADING_EDGE = 1
LABEL_TRAILING_EDGE = 2
LABEL_TIP = 3

MESH_DENSITIES = {
    "coarse": (10, 7),
    "medium": (14, 10),
    "fine": (18, 13),
}
MESHERS = (
    "structured_alternating",
    "delaunay_uniform",
    "delaunay_random",
    "delaunay_anisotropic",
)


def naca_00xx_thickness(x_fraction: np.ndarray, thickness: float = 0.12) -> np.ndarray:
    x = np.clip(x_fraction, 0.0, 1.0)
    yt = 5 * thickness * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )
    return yt


def chord_at_y(params: WingParameters, y: np.ndarray) -> np.ndarray:
    eta = np.abs(y) / (params.span / 2)
    if params.planform_family == "elliptical":
        ellipse = np.sqrt(np.maximum(1.0 - eta**2, 0.0))
        return params.tip_chord + (params.root_chord - params.tip_chord) * ellipse
    if params.planform_family == "cranked":
        kink = params.kink_span_fraction
        kink_chord = params.root_chord * params.kink_chord_ratio
        inner = params.root_chord + (kink_chord - params.root_chord) * eta / max(kink, 1e-9)
        outer = kink_chord + (params.tip_chord - kink_chord) * (eta - kink) / max(1.0 - kink, 1e-9)
        return np.where(eta <= kink, inner, outer)
    return params.root_chord + (params.tip_chord - params.root_chord) * eta


def leading_edge_x(params: WingParameters, y: np.ndarray) -> np.ndarray:
    eta = np.abs(y) / (params.span / 2)
    return np.tan(np.deg2rad(params.sweep_deg)) * np.abs(y) + 0.0 * eta


def _camber_z(params: WingParameters, y: np.ndarray, x_fraction: np.ndarray) -> np.ndarray:
    dihedral = np.tan(np.deg2rad(params.dihedral_deg)) * np.abs(y)
    twist = np.deg2rad(params.twist_deg) * (np.abs(y) / (params.span / 2))
    chord = chord_at_y(params, y)
    camber = 4.0 * params.camber_ratio * x_fraction * (1.0 - x_fraction) * chord
    # Positive twist is wash-in. With freestream z positive at positive alpha,
    # the trailing edge therefore moves towards negative z.
    return dihedral + camber - np.tan(twist) * (x_fraction - 0.25) * chord


def aerodynamic_reference(params: WingParameters, samples: int = 10001) -> tuple[float, float, float, np.ndarray]:
    """Return Sref, Cref, Bref and the quarter-MAC reference point."""
    y = np.linspace(-0.5 * params.span, 0.5 * params.span, samples)
    chord = chord_at_y(params, y)
    leading = leading_edge_x(params, y)
    area = float(np.trapezoid(chord, y))
    mean_aerodynamic_chord = float(np.trapezoid(chord**2, y) / max(area, 1e-12))
    quarter_chord_x = float(np.trapezoid((leading + 0.25 * chord) * chord, y) / max(area, 1e-12))
    reference_point = np.array([quarter_chord_x, 0.0, 0.0], dtype=float)
    return area, mean_aerodynamic_chord, params.span, reference_point


def generate_wing_surface(
    params: WingParameters,
    n_span_surface: int | None = None,
    n_chord_surface: int | None = None,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a simple triangulated wing surface with ground-truth feature labels."""
    rng = np.random.default_rng(seed)
    ns = n_span_surface or max(params.n_span * 2, 16)
    nc = n_chord_surface or max(params.n_chord * 4, 16)
    ys = np.linspace(-params.span / 2, params.span / 2, ns + 1)
    # Cosine spacing resolves the rounded leading edge without requiring a
    # a separate remeshing dependency.
    xs = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, nc + 1)))

    vertices: list[list[float]] = []
    labels: list[int] = []
    index: dict[tuple[int, int, int], int] = {}
    for side in [1, -1]:
        for i, y in enumerate(ys):
            chord = chord_at_y(params, np.array([y]))[0]
            le = leading_edge_x(params, np.array([y]))[0]
            for j, xf in enumerate(xs):
                z = _camber_z(params, np.array([y]), np.array([xf]))[0]
                z += side * naca_00xx_thickness(np.array([xf]), params.thickness_ratio)[0] * chord
                pt = np.array([le + xf * chord, y, z], dtype=float)
                index[(side, i, j)] = len(vertices)
                vertices.append(pt.tolist())
                if j == 0:
                    label = LABEL_LEADING_EDGE
                elif j == nc:
                    label = LABEL_TRAILING_EDGE
                elif i in (0, ns):
                    label = LABEL_TIP
                else:
                    label = LABEL_SURFACE
                labels.append(label)

    faces: list[list[int]] = []
    for side in [1, -1]:
        for i in range(ns):
            for j in range(nc):
                a = index[(side, i, j)]
                b = index[(side, i + 1, j)]
                c = index[(side, i + 1, j + 1)]
                d = index[(side, i, j + 1)]
                if side == 1:
                    faces.append([a, b, c])
                    faces.append([a, c, d])
                else:
                    faces.append([a, c, b])
                    faces.append([a, d, c])

    # Close the perimeter so the generated surface behaves like an STL solid.
    for i in range(ns):
        for j in (0, nc):
            upper_a = index[(1, i, j)]
            upper_b = index[(1, i + 1, j)]
            lower_a = index[(-1, i, j)]
            lower_b = index[(-1, i + 1, j)]
            faces.append([upper_a, lower_b, upper_b])
            faces.append([upper_a, lower_a, lower_b])
    for i in (0, ns):
        for j in range(nc):
            upper_a = index[(1, i, j)]
            upper_b = index[(1, i, j + 1)]
            lower_a = index[(-1, i, j)]
            lower_b = index[(-1, i, j + 1)]
            faces.append([upper_a, upper_b, lower_b])
            faces.append([upper_a, lower_b, lower_a])
    vertex_array = np.array(vertices, dtype=float)
    face_array = np.array(faces, dtype=int)
    label_array = np.array(labels, dtype=int)

    # Upper and lower surfaces share the geometric leading/trailing edges.
    # Merge those vertices before perturbation so generated STL cases remain
    # watertight and retain coherent edge topology under noise.
    unique_vertices, inverse = np.unique(np.round(vertex_array, 12), axis=0, return_inverse=True)
    unique_labels = np.zeros(len(unique_vertices), dtype=int)
    for original_index, unique_index in enumerate(inverse):
        unique_labels[unique_index] = max(unique_labels[unique_index], label_array[original_index])
    vertex_array = unique_vertices.astype(float)
    face_array = inverse[face_array]
    if noise:
        vertex_array += rng.normal(scale=noise, size=vertex_array.shape)
    triangles = vertex_array[face_array]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    return vertex_array, face_array[double_areas > 1e-12], unique_labels


def random_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    q *= signs
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def generate_wing_trimesh(
    params: WingParameters,
    noise: float = 0.0,
    seed: int = 0,
    rotate: bool = False,
) -> TriMesh:
    vertices, faces, labels = generate_wing_surface(params, noise=noise, seed=seed)
    if rotate:
        vertices = vertices @ random_rotation(seed + 7919)
        vertices += np.random.default_rng(seed + 104729).uniform(-2.0, 2.0, size=3)
    return TriMesh(
        vertices,
        faces,
        labels,
        metadata={"name": params.name, "params": params.__dict__.copy()},
    )


def _structured_normalized_triangulation(
    n_span: int,
    n_chord: int,
) -> tuple[np.ndarray, np.ndarray]:
    eta = np.linspace(-1.0, 1.0, n_span + 1)
    xi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_chord + 1)))
    uv = np.array([[span, chord] for span in eta for chord in xi], dtype=float)
    faces: list[list[int]] = []

    def index(i: int, j: int) -> int:
        return i * (n_chord + 1) + j

    for i in range(n_span):
        for j in range(n_chord):
            a, b = index(i, j), index(i + 1, j)
            c, d = index(i + 1, j + 1), index(i, j + 1)
            if (i + j) % 2:
                faces.extend(([a, b, d], [b, c, d]))
            else:
                faces.extend(([a, b, c], [a, c, d]))
    return uv, np.asarray(faces, dtype=int)


def _delaunay_normalized_triangulation(
    n_span: int,
    n_chord: int,
    mesher: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    boundary_eta = np.linspace(-1.0, 1.0, n_span + 1)
    boundary_xi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_chord + 1)))
    boundary = np.vstack(
        [
            np.column_stack([boundary_eta, np.zeros_like(boundary_eta)]),
            np.column_stack([boundary_eta, np.ones_like(boundary_eta)]),
            np.column_stack([np.full_like(boundary_xi, -1.0), boundary_xi]),
            np.column_stack([np.full_like(boundary_xi, 1.0), boundary_xi]),
        ]
    )
    interior_count = max((n_span - 1) * (n_chord - 1), 1)
    if mesher == "delaunay_uniform":
        eta = np.linspace(-1.0, 1.0, n_span + 1)[1:-1]
        xi = boundary_xi[1:-1]
        interior = np.array([[span, chord] for span in eta for chord in xi], dtype=float)
        # Break co-circular ties without changing the represented surface.
        spacing = min(2.0 / n_span, 1.0 / n_chord)
        interior += rng.normal(0.0, 0.015 * spacing, size=interior.shape)
    elif mesher == "delaunay_random":
        interior = np.column_stack(
            [rng.uniform(-1.0, 1.0, interior_count), rng.uniform(0.0, 1.0, interior_count)]
        )
    elif mesher == "delaunay_anisotropic":
        raw_eta = rng.uniform(-1.0, 1.0, interior_count)
        raw_xi = rng.uniform(-1.0, 1.0, interior_count)
        eta = np.sign(raw_eta) * np.abs(raw_eta) ** 1.65
        xi = 0.5 * (1.0 + np.sign(raw_xi) * np.abs(raw_xi) ** 0.58)
        interior = np.column_stack([eta, xi])
    else:
        raise ValueError(f"unsupported Delaunay mesher: {mesher}")
    uv = np.unique(np.round(np.vstack([boundary, interior]), 12), axis=0)
    faces = Delaunay(uv).simplices.astype(int)
    return uv, faces


def _map_normalized_surface(
    params: WingParameters,
    uv: np.ndarray,
    side: int,
) -> np.ndarray:
    y = 0.5 * params.span * uv[:, 0]
    fraction = uv[:, 1]
    chord = chord_at_y(params, y)
    leading = leading_edge_x(params, y)
    mean = _camber_z(params, y, fraction)
    thickness = naca_00xx_thickness(fraction, params.thickness_ratio) * chord
    return np.column_stack([leading + fraction * chord, y, mean + side * thickness])


def _orient_surface_faces(vertices: np.ndarray, faces: np.ndarray, upward: bool) -> np.ndarray:
    oriented = faces.copy()
    triangles = vertices[oriented]
    normal_z = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[:, 2]
    reverse = normal_z < 0.0 if upward else normal_z > 0.0
    oriented[reverse] = oriented[reverse][:, [0, 2, 1]]
    return oriented


def generate_remeshed_wing_trimesh(
    params: WingParameters,
    mesher: str,
    mesh_density: str,
    seed: int,
    rotate: bool = True,
) -> TriMesh:
    """Triangulate one continuous lifting surface independently of its geometry."""
    if mesh_density not in MESH_DENSITIES:
        raise ValueError(f"unknown mesh density: {mesh_density}")
    if mesher not in MESHERS:
        raise ValueError(f"unknown mesher: {mesher}")
    n_span, n_chord = MESH_DENSITIES[mesh_density]
    if mesher == "structured_alternating":
        uv, surface_faces = _structured_normalized_triangulation(n_span, n_chord)
    else:
        uv, surface_faces = _delaunay_normalized_triangulation(n_span, n_chord, mesher, seed)

    upper = _map_normalized_surface(params, uv, 1)
    lower = _map_normalized_surface(params, uv, -1)
    upper_faces = _orient_surface_faces(upper, surface_faces, upward=True)
    lower_faces = _orient_surface_faces(lower, surface_faces, upward=False) + len(uv)
    vertices = np.vstack([upper, lower])
    faces = np.vstack([upper_faces, lower_faces])

    # Close both tips. Leading and trailing edges require no cap because the
    # section thickness is exactly zero there.
    tip_faces: list[list[int]] = []
    for eta_value in (-1.0, 1.0):
        indices = np.flatnonzero(np.isclose(uv[:, 0], eta_value, atol=1e-12))
        indices = indices[np.argsort(uv[indices, 1])]
        for first, second in zip(indices[:-1], indices[1:]):
            upper_first, upper_second = int(first), int(second)
            lower_first, lower_second = int(first + len(uv)), int(second + len(uv))
            candidate = np.asarray(
                [[upper_first, upper_second, lower_second], [upper_first, lower_second, lower_first]],
                dtype=int,
            )
            desired_y = -1.0 if eta_value < 0.0 else 1.0
            triangles = vertices[candidate]
            normal_y = np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            )[:, 1]
            reverse = normal_y * desired_y < 0.0
            candidate[reverse] = candidate[reverse][:, [0, 2, 1]]
            tip_faces.extend(candidate.tolist())
    faces = np.vstack([faces, np.asarray(tip_faces, dtype=int)])

    # Weld the coincident leading/trailing vertices with a geometry-relative tolerance.
    extent = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1.0)
    tolerance = 1e-10 * extent
    keys = np.round(vertices / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    welded_vertices = vertices[first]
    welded_faces = inverse[faces]
    nondegenerate = np.array([len(set(face)) == 3 for face in welded_faces], dtype=bool)
    welded_faces = welded_faces[nondegenerate]
    triangles = welded_vertices[welded_faces]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    welded_faces = welded_faces[double_area > tolerance**2]

    labels = np.zeros(len(welded_vertices), dtype=int)
    source_uv = np.vstack([uv, uv])
    for old_index, new_index in enumerate(inverse):
        eta, fraction = source_uv[old_index]
        if np.isclose(fraction, 0.0, atol=1e-12):
            label = LABEL_LEADING_EDGE
        elif np.isclose(fraction, 1.0, atol=1e-12):
            label = LABEL_TRAILING_EDGE
        elif np.isclose(abs(eta), 1.0, atol=1e-12):
            label = LABEL_TIP
        else:
            label = LABEL_SURFACE
        labels[new_index] = max(labels[new_index], label)

    source_edge_labels: dict[tuple[int, int], int] = {}
    representative_old = np.full(len(welded_vertices), -1, dtype=int)
    for old_index, new_index in enumerate(inverse):
        if representative_old[new_index] < 0:
            representative_old[new_index] = old_index
    for face in welded_faces:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(start), int(end))))
            old_start = int(representative_old[edge[0]])
            old_end = int(representative_old[edge[1]])
            uv_start, uv_end = source_uv[old_start], source_uv[old_end]
            label = LABEL_SURFACE
            if np.isclose(uv_start[1], 0.0, atol=1e-12) and np.isclose(uv_end[1], 0.0, atol=1e-12):
                label = LABEL_LEADING_EDGE
            elif np.isclose(uv_start[1], 1.0, atol=1e-12) and np.isclose(uv_end[1], 1.0, atol=1e-12):
                label = LABEL_TRAILING_EDGE
            elif np.isclose(abs(uv_start[0]), 1.0, atol=1e-12) and np.isclose(
                uv_start[0], uv_end[0], atol=1e-12
            ):
                label = LABEL_TIP
            source_edge_labels[edge] = max(source_edge_labels.get(edge, 0), label)

    curve_eta = np.linspace(-1.0, 1.0, 401)
    leading_curve = _map_normalized_surface(
        params, np.column_stack([curve_eta, np.zeros_like(curve_eta)]), 1
    )
    trailing_curve = _map_normalized_surface(
        params, np.column_stack([curve_eta, np.ones_like(curve_eta)]), 1
    )
    curve_xi = np.linspace(0.0, 1.0, 201)
    tip_curves = [
        _map_normalized_surface(
            params, np.column_stack([np.full_like(curve_xi, eta), curve_xi]), 1
        )
        for eta in (-1.0, 1.0)
    ]
    tip_boundary_options = [
        [
            _map_normalized_surface(
                params,
                np.column_stack([np.full_like(curve_xi, eta), curve_xi]),
                side,
            )
            for side in (1, -1)
        ]
        for eta in (-1.0, 1.0)
    ]

    if rotate:
        rotation = random_rotation(seed + 7919)
        translation = np.random.default_rng(seed + 104729).uniform(-2.0, 2.0, size=3)
        welded_vertices = welded_vertices @ rotation + translation
        leading_curve = leading_curve @ rotation + translation
        trailing_curve = trailing_curve @ rotation + translation
        tip_curves = [curve @ rotation + translation for curve in tip_curves]
        tip_boundary_options = [
            [curve @ rotation + translation for curve in options]
            for options in tip_boundary_options
        ]

    return TriMesh(
        welded_vertices,
        welded_faces,
        labels,
        metadata={
            "name": params.name,
            "params": params.__dict__.copy(),
            "mesher": mesher,
            "mesh_density": mesh_density,
            "remesh_seed": seed,
            "source_kind": "synthetic_parametric",
            "source_edge_labels": source_edge_labels,
            "source_curves": {
                "leading_edge": leading_curve,
                "trailing_edge": trailing_curve,
                "tips": tip_curves,
            },
            "source_tip_boundary_options": tip_boundary_options,
            "source_root_chord": params.root_chord,
        },
    )


def extract_geometry_features(vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray | None = None) -> GeometryFeatures:
    face_normals = np.zeros((len(faces), 3), dtype=float)
    vertex_normals = np.zeros_like(vertices)
    incident: list[list[int]] = [[] for _ in range(len(vertices))]

    for idx, tri in enumerate(faces):
        a, b, c = vertices[tri]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            n = n / norm
        face_normals[idx] = n
        for v in tri:
            vertex_normals[v] += n
            incident[v].append(idx)

    norms = np.linalg.norm(vertex_normals, axis=1)
    vertex_normals[norms > 1e-12] /= norms[norms > 1e-12, None]

    saliency = np.zeros(len(vertices), dtype=float)
    for i, faces_i in enumerate(incident):
        if len(faces_i) < 2:
            continue
        local = face_normals[faces_i]
        mean = local.mean(axis=0)
        saliency[i] = float(np.mean(np.linalg.norm(local - mean, axis=1)))

    x = vertices[:, 0]
    y = vertices[:, 1]
    chord_fraction = (x - x.min()) / max(x.max() - x.min(), 1e-9)
    span_fraction = (y - y.min()) / max(y.max() - y.min(), 1e-9)
    return GeometryFeatures(vertices, faces, vertex_normals, saliency, chord_fraction, span_fraction, labels)


def feature_matrix(features: GeometryFeatures) -> np.ndarray:
    return np.column_stack(
        [
            features.vertices,
            features.vertex_normals,
            features.saliency,
            features.chord_fraction,
            features.span_fraction,
        ]
    )


def rule_based_labels(features: GeometryFeatures) -> np.ndarray:
    labels = np.full(len(features.vertices), LABEL_SURFACE, dtype=int)
    labels[features.chord_fraction < 0.08] = LABEL_LEADING_EDGE
    labels[features.chord_fraction > 0.92] = LABEL_TRAILING_EDGE
    labels[(features.span_fraction < 0.03) | (features.span_fraction > 0.97)] = LABEL_TIP
    return labels


def make_wing_params_for_aspect_ratio(
    aspect_ratio: float,
    name: str | None = None,
    area: float = 16.0,
    sweep_deg: float = 0.0,
    taper_ratio: float = 1.0,
    n_span: int = 16,
    n_chord: int = 4,
    planform_family: str = "trapezoidal",
    kink_span_fraction: float = 0.45,
    kink_chord_ratio: float = 0.72,
) -> WingParameters:
    span = math.sqrt(aspect_ratio * area)
    if planform_family == "elliptical":
        area_per_root_chord = span * (taper_ratio + (1.0 - taper_ratio) * math.pi / 4.0)
    elif planform_family == "cranked":
        half_span = 0.5 * span
        kink_span = kink_span_fraction * half_span
        half_area_per_root = 0.5 * (1.0 + kink_chord_ratio) * kink_span
        half_area_per_root += 0.5 * (kink_chord_ratio + taper_ratio) * (half_span - kink_span)
        area_per_root_chord = 2.0 * half_area_per_root
    else:
        area_per_root_chord = 0.5 * span * (1.0 + taper_ratio)
    root_chord = area / max(area_per_root_chord, 1e-12)
    tip_chord = root_chord * taper_ratio
    return WingParameters(
        name=name or f"AR{aspect_ratio:g}",
        span=span,
        root_chord=root_chord,
        tip_chord=tip_chord,
        sweep_deg=sweep_deg,
        planform_family=planform_family,
        kink_span_fraction=kink_span_fraction,
        kink_chord_ratio=kink_chord_ratio,
        n_span=n_span,
        n_chord=n_chord,
    )


def generate_aero_mesh(params: WingParameters) -> AeroMesh:
    ys = np.linspace(-params.span / 2, params.span / 2, params.n_span + 1)
    xfs = np.linspace(0.0, 1.0, params.n_chord + 1)
    grid = np.zeros((params.n_span + 1, params.n_chord + 1, 3), dtype=float)
    for i, y in enumerate(ys):
        chord = chord_at_y(params, np.array([y]))[0]
        le = leading_edge_x(params, np.array([y]))[0]
        for j, xf in enumerate(xfs):
            grid[i, j] = [le + xf * chord, y, _camber_z(params, np.array([y]), np.array([xf]))[0]]

    vertices = grid.reshape((-1, 3))
    panels: list[list[int]] = []
    collocation: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    bound_left: list[np.ndarray] = []
    bound_right: list[np.ndarray] = []
    span_widths: list[float] = []
    wake_edges: list[list[int]] = []

    def idx(i: int, j: int) -> int:
        return i * (params.n_chord + 1) + j

    for i in range(params.n_span):
        for j in range(params.n_chord):
            panel = [idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
            panels.append(panel)
            p00, p10, p11, p01 = vertices[panel]
            chord_left = p01 - p00
            chord_right = p11 - p10
            span_front = p10 - p00
            normal = np.cross(chord_left + chord_right, span_front)
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
            if normal[2] < 0:
                normal = -normal
            normals.append(normal)
            collocation.append(0.125 * (p00 + p10) + 0.375 * (p01 + p11))
            bound_left.append(0.75 * p00 + 0.25 * p01)
            bound_right.append(0.75 * p10 + 0.25 * p11)
            span_widths.append(float(np.linalg.norm(p10 - p00)))
            if j == params.n_chord - 1:
                wake_edges.append([len(panels) - 1, panel[1], panel[2]])

    reference_area, reference_chord, reference_span, reference_point = aerodynamic_reference(params)
    return AeroMesh(
        vertices=vertices,
        panels=np.array(panels, dtype=int),
        collocation_points=np.array(collocation, dtype=float),
        normals=np.array(normals, dtype=float),
        bound_left=np.array(bound_left, dtype=float),
        bound_right=np.array(bound_right, dtype=float),
        span_widths=np.array(span_widths, dtype=float),
        reference_area=reference_area,
        reference_chord=reference_chord,
        reference_span=reference_span,
        reference_point=reference_point,
        wake_edges=np.array(wake_edges, dtype=int),
        body_axes=np.eye(3),
        metadata={"params": params.__dict__, "n_span": params.n_span, "n_chord": params.n_chord},
    )


def transform_aero_mesh(
    mesh: AeroMesh,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    scale: float = 1.0,
) -> AeroMesh:
    """Apply a proper rigid-body transform and uniform scale."""
    transform = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    if transform.shape != (3, 3) or not np.allclose(transform.T @ transform, np.eye(3), atol=1e-10):
        raise ValueError("rotation must be an orthogonal 3 by 3 matrix")
    if np.linalg.det(transform) < 0.0:
        raise ValueError("rotation must preserve orientation")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    offset = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)

    def points(values: np.ndarray) -> np.ndarray:
        return scale * (values @ transform.T) + offset

    def vectors(values: np.ndarray) -> np.ndarray:
        return values @ transform.T

    return AeroMesh(
        vertices=points(mesh.vertices),
        panels=mesh.panels.copy(),
        collocation_points=points(mesh.collocation_points),
        normals=vectors(mesh.normals),
        bound_left=points(mesh.bound_left),
        bound_right=points(mesh.bound_right),
        span_widths=scale * mesh.span_widths,
        reference_area=scale**2 * mesh.reference_area,
        reference_chord=scale * mesh.reference_chord,
        reference_span=scale * mesh.reference_span,
        reference_point=points(mesh.reference_point[None, :])[0],
        wake_edges=mesh.wake_edges.copy(),
        body_axes=transform @ mesh.body_axes,
        metadata={**mesh.metadata, "transformed": True},
    )


def recover_mesh_from_labels(
    params: WingParameters,
    features: GeometryFeatures,
    predicted_labels: np.ndarray,
) -> tuple[AeroMesh, bool]:
    """Recover a structured mesh from classified surface features.

    The first version keeps the parametric topology but uses the classifier
    output as a validity gate for leading/trailing-edge recovery.
    """
    le_count = int(np.sum(predicted_labels == LABEL_LEADING_EDGE))
    te_count = int(np.sum(predicted_labels == LABEL_TRAILING_EDGE))
    tip_count = int(np.sum(predicted_labels == LABEL_TIP))
    required = max(4, params.n_span // 2)
    valid = le_count >= required and te_count >= required and tip_count >= 2
    mesh = generate_aero_mesh(replace(params))
    mesh.metadata.update(
        {
            "feature_counts": {
                "leading_edge": le_count,
                "trailing_edge": te_count,
                "tip": tip_count,
            },
            "classification_valid": valid,
        }
    )
    return mesh, valid


def panel_quality(mesh: AeroMesh) -> dict[str, float]:
    areas = []
    aspect_ratios = []
    for panel in mesh.panels:
        pts = mesh.vertices[panel]
        area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[3] - pts[0]))
        area += 0.5 * np.linalg.norm(np.cross(pts[2] - pts[1], pts[3] - pts[1]))
        widths = [
            np.linalg.norm(pts[(i + 1) % 4] - pts[i])
            for i in range(4)
        ]
        min_w = max(min(widths), 1e-9)
        aspect_ratios.append(max(widths) / min_w)
        areas.append(area)
    areas_a = np.array(areas)
    ar_a = np.array(aspect_ratios)
    return {
        "min_area": float(areas_a.min()),
        "mean_area": float(areas_a.mean()),
        "max_panel_aspect_ratio": float(ar_a.max()),
        "valid_panel_fraction": float(np.mean(areas_a > 1e-10)),
    }
