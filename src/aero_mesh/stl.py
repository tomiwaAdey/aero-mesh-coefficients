from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .types import CanonicalFrame, TriMesh


EDGE_FEATURE_NAMES = (
    "midpoint_x",
    "midpoint_y",
    "midpoint_z",
    "direction_x",
    "direction_y",
    "direction_z",
    "relative_length",
    "dihedral",
    "boundary",
    "local_chord_fraction",
    "absolute_span_fraction",
    "spanwise_direction",
    "normal_x",
    "normal_y",
    "normal_z",
    "normal_variation",
    "pca_surface_variation",
    "pca_linearity",
    "pca_planarity",
)


def _face_edge_uses(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    uses: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, face in enumerate(faces):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            ordered = (int(start), int(end))
            edge = tuple(sorted(ordered))
            direction = 1 if ordered == edge else -1
            uses.setdefault(edge, []).append((face_index, direction))
    return uses


def _orient_faces_componentwise(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, bool]:
    oriented = faces.copy()
    uses = _face_edge_uses(oriented)
    neighbours: list[list[tuple[int, bool]]] = [[] for _ in range(len(oriented))]
    for edge_uses in uses.values():
        if len(edge_uses) != 2:
            continue
        (first, first_direction), (second, second_direction) = edge_uses
        must_differ = first_direction == second_direction
        neighbours[first].append((second, must_differ))
        neighbours[second].append((first, must_differ))

    flips = np.full(len(oriented), -1, dtype=int)
    components: list[list[int]] = []
    consistent = True
    for start in range(len(oriented)):
        if flips[start] >= 0:
            continue
        flips[start] = 0
        component = []
        stack = [start]
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour, must_differ in neighbours[current]:
                expected = flips[current] ^ int(must_differ)
                if flips[neighbour] < 0:
                    flips[neighbour] = expected
                    stack.append(neighbour)
                elif flips[neighbour] != expected:
                    consistent = False
        components.append(component)
    oriented[flips == 1] = oriented[flips == 1][:, [0, 2, 1]]

    oriented_uses = _face_edge_uses(oriented)
    for edge_uses in oriented_uses.values():
        if len(edge_uses) == 2 and edge_uses[0][1] == edge_uses[1][1]:
            consistent = False

    # Give each closed component an outward orientation.
    for component in components:
        component_set = set(component)
        component_closed = all(
            len(edge_uses) == 2
            for edge_uses in oriented_uses.values()
            if any(face_index in component_set for face_index, _ in edge_uses)
        )
        if not component_closed:
            continue
        triangles = vertices[oriented[component]]
        signed_volume = np.sum(
            np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
        ) / 6.0
        if signed_volume < 0.0:
            oriented[component] = oriented[component][:, [0, 2, 1]]
    return oriented, consistent


def repair_trimesh(
    mesh: TriMesh,
    weld_tolerance_relative: float = 1e-9,
    intentionally_open: bool | None = None,
) -> TriMesh:
    """Weld, clean and coherently orient a triangulated surface."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    diagonal = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
    tolerance = max(weld_tolerance_relative * diagonal, np.finfo(float).eps * diagonal * 16.0)
    keys = np.round(vertices / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    welded_vertices = vertices[first]
    remapped = inverse[faces]

    unique_vertex_faces = np.array([len(set(face.tolist())) == 3 for face in remapped], dtype=bool)
    removed_collapsed = int(np.sum(~unique_vertex_faces))
    remapped = remapped[unique_vertex_faces]
    triangles = welded_vertices[remapped]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    area_tolerance = max(diagonal**2 * 1e-14, tolerance**2)
    nonzero = double_areas > area_tolerance
    removed_zero_area = int(np.sum(~nonzero))
    remapped = remapped[nonzero]

    canonical_faces = np.sort(remapped, axis=1)
    _, unique_face_indices = np.unique(canonical_faces, axis=0, return_index=True)
    unique_face_indices = np.sort(unique_face_indices)
    removed_duplicates = int(len(remapped) - len(unique_face_indices))
    remapped = remapped[unique_face_indices]
    remapped, orientation_consistent = _orient_faces_componentwise(welded_vertices, remapped)

    labels = None
    if mesh.labels is not None:
        labels = np.zeros(len(welded_vertices), dtype=int)
        np.maximum.at(labels, inverse, np.asarray(mesh.labels, dtype=int))

    metadata = dict(mesh.metadata)
    if "source_edge_labels" in metadata:
        remapped_labels: dict[tuple[int, int], int] = {}
        for edge, label in metadata["source_edge_labels"].items():
            mapped = tuple(sorted((int(inverse[edge[0]]), int(inverse[edge[1]]))))
            if mapped[0] != mapped[1]:
                remapped_labels[mapped] = max(remapped_labels.get(mapped, 0), int(label))
        metadata["source_edge_labels"] = remapped_labels

    edge_uses = _face_edge_uses(remapped)
    boundary_edges = sum(len(uses) == 1 for uses in edge_uses.values())
    nonmanifold_edges = sum(len(uses) > 2 for uses in edge_uses.values())
    closed = bool(edge_uses) and boundary_edges == 0 and nonmanifold_edges == 0
    metadata.update(
        {
            "weld_tolerance": tolerance,
            "removed_collapsed_faces": removed_collapsed,
            "removed_zero_area_faces": removed_zero_area,
            "removed_duplicate_faces": removed_duplicates,
            "boundary_edges": boundary_edges,
            "nonmanifold_edges": nonmanifold_edges,
            "closed": closed,
            "consistently_oriented": bool(orientation_consistent),
            "intentionally_open": bool(not closed if intentionally_open is None else intentionally_open),
        }
    )
    return TriMesh(welded_vertices, remapped.astype(int), labels, metadata)


def _deduplicate(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    repaired = repair_trimesh(TriMesh(vertices, faces))
    return repaired.vertices, repaired.faces


def load_stl(path: str | Path) -> TriMesh:
    """Load binary or ASCII STL without relying on a CAD package."""
    stl_path = Path(path)
    raw = stl_path.read_bytes()
    if len(raw) >= 84:
        triangle_count = struct.unpack_from("<I", raw, 80)[0]
        if 84 + triangle_count * 50 == len(raw):
            vertices = np.empty((triangle_count * 3, 3), dtype=float)
            faces = np.arange(triangle_count * 3, dtype=int).reshape((-1, 3))
            offset = 84
            for index in range(triangle_count):
                values = struct.unpack_from("<12fH", raw, offset)
                vertices[index * 3 : index * 3 + 3] = np.array(values[3:12], dtype=float).reshape((3, 3))
                offset += 50
            return repair_trimesh(
                TriMesh(vertices, faces, metadata={"source": str(stl_path), "format": "binary-stl"}),
                intentionally_open=False,
            )

    text = raw.decode("utf-8", errors="ignore")
    points = []
    for line in text.splitlines():
        tokens = line.strip().split()
        if len(tokens) == 4 and tokens[0].lower() == "vertex":
            points.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
    if not points or len(points) % 3:
        raise ValueError(f"Invalid STL geometry: {stl_path}")
    vertices = np.asarray(points, dtype=float)
    faces = np.arange(len(vertices), dtype=int).reshape((-1, 3))
    return repair_trimesh(
        TriMesh(vertices, faces, metadata={"source": str(stl_path), "format": "ascii-stl"}),
        intentionally_open=False,
    )


def load_gmsh_v2_surface(
    path: str | Path,
    physical_tag: int | None = None,
    entity_tags: set[int] | None = None,
) -> TriMesh:
    """Load selected triangular surfaces from an ASCII Gmsh 2.2 file."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        iterator = iter(handle)
        for line in iterator:
            if line.strip() == "$Nodes":
                break
        else:
            raise ValueError("Gmsh file does not contain a $Nodes section")

        node_count = int(next(iterator).strip())
        nodes: dict[int, np.ndarray] = {}
        for _ in range(node_count):
            row = next(iterator).split()
            nodes[int(row[0])] = np.asarray(row[1:4], dtype=float)

        for line in iterator:
            if line.strip() == "$Elements":
                break
        else:
            raise ValueError("Gmsh file does not contain an $Elements section")

        element_count = int(next(iterator).strip())
        triangle_nodes: list[tuple[int, int, int]] = []
        for _ in range(element_count):
            row = [int(value) for value in next(iterator).split()]
            element_type = row[1]
            tag_count = row[2]
            tags = row[3 : 3 + tag_count]
            connectivity = row[3 + tag_count :]
            physical_matches = physical_tag is None or (tags and tags[0] == physical_tag)
            entity_matches = entity_tags is None or (len(tags) > 1 and tags[1] in entity_tags)
            if element_type == 2 and physical_matches and entity_matches:
                triangle_nodes.append(tuple(connectivity[:3]))

    if not triangle_nodes:
        raise ValueError("No triangular elements matched the requested Gmsh tags")
    used_nodes = sorted({node for triangle in triangle_nodes for node in triangle})
    remap = {node: index for index, node in enumerate(used_nodes)}
    vertices = np.vstack([nodes[node] for node in used_nodes])
    faces = np.asarray([[remap[node] for node in triangle] for triangle in triangle_nodes], dtype=int)
    return repair_trimesh(
        TriMesh(
            vertices,
            faces,
            metadata={
            "source": str(source),
            "physical_tag": physical_tag,
            "entity_tags": None if entity_tags is None else sorted(entity_tags),
            },
        ),
        intentionally_open=True,
    )


def mirror_half_surface(mesh: TriMesh, axis: int) -> TriMesh:
    """Mirror a half-model and weld vertices on its symmetry plane."""
    mirrored = mesh.vertices.copy()
    mirrored[:, axis] *= -1.0
    vertices = np.vstack([mesh.vertices, mirrored])
    offset = len(mesh.vertices)
    faces = np.vstack([mesh.faces, mesh.faces[:, ::-1] + offset])
    labels = None
    if mesh.labels is not None:
        labels = np.concatenate([mesh.labels, mesh.labels])
    metadata = {**mesh.metadata, "mirrored_axis": axis}
    source_edge_labels = mesh.metadata.get("source_edge_labels")
    if source_edge_labels is not None:
        doubled_edge_labels: dict[tuple[int, int], int] = {}
        for edge, label in source_edge_labels.items():
            original = tuple(sorted((int(edge[0]), int(edge[1]))))
            reflected = tuple(sorted((int(edge[0]) + offset, int(edge[1]) + offset)))
            doubled_edge_labels[original] = int(label)
            doubled_edge_labels[reflected] = int(label)
        metadata["source_edge_labels"] = doubled_edge_labels
    source_curves = mesh.metadata.get("source_curves")
    if source_curves is not None:
        leading = np.asarray(source_curves["leading_edge"], dtype=float)
        trailing = np.asarray(source_curves["trailing_edge"], dtype=float)
        leading_mirror = leading.copy()
        trailing_mirror = trailing.copy()
        leading_mirror[:, axis] *= -1.0
        trailing_mirror[:, axis] *= -1.0
        tips = [np.asarray(curve, dtype=float) for curve in source_curves["tips"]]
        outer_tip = tips[-1]
        mirrored_tip = outer_tip.copy()
        mirrored_tip[:, axis] *= -1.0
        metadata["source_curves"] = {
            "leading_edge": np.vstack([leading_mirror[::-1], leading[1:]]),
            "trailing_edge": np.vstack([trailing_mirror[::-1], trailing[1:]]),
            "tips": [mirrored_tip, outer_tip],
        }
    return repair_trimesh(
        TriMesh(
            vertices,
            faces,
            labels,
            metadata=metadata,
        ),
        intentionally_open=bool(mesh.metadata.get("intentionally_open", False)),
    )


def write_binary_stl(path: str | Path, mesh: TriMesh) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = b"aero-mesh-coefficients".ljust(80, b" ")
    payload = bytearray(header)
    payload.extend(struct.pack("<I", len(mesh.faces)))
    for face in mesh.faces:
        a, b, c = mesh.vertices[face]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
        payload.extend(struct.pack("<12fH", *(normal.tolist() + a.tolist() + b.tolist() + c.tolist()), 0))
    target.write_bytes(payload)
    return target


def apply_frame(vertices: np.ndarray, frame: CanonicalFrame) -> np.ndarray:
    return ((vertices - frame.origin) @ frame.axes) / frame.scale


def _surface_area_moments(mesh: TriMesh) -> tuple[np.ndarray, np.ndarray]:
    """Return exact first and second moments of the triangulated surface."""
    triangles = mesh.vertices[mesh.faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    valid = areas > 1e-15
    if not np.any(valid):
        raise ValueError("mesh has no finite-area faces")
    triangles = triangles[valid]
    areas = areas[valid]
    centroid = np.average(triangles.mean(axis=1), axis=0, weights=areas)

    vertex_sum = triangles.sum(axis=1)
    sum_outer = np.einsum("fi,fj->fij", vertex_sum, vertex_sum)
    vertex_outer = np.einsum("fvi,fvj->fij", triangles, triangles)
    second_moment = np.average(
        (sum_outer + vertex_outer) / 12.0,
        axis=0,
        weights=areas,
    )
    covariance = second_moment - np.outer(centroid, centroid)
    return centroid, 0.5 * (covariance + covariance.T)


def _closure_chord_axis(
    mesh: TriMesh,
    plane_origin: np.ndarray,
    span_axis: np.ndarray,
    section_basis: np.ndarray,
) -> np.ndarray | None:
    """Find the root chord from upper/lower closure edges crossing the root plane."""
    edges, adjacent_faces = edge_topology(mesh)
    normals = face_normals(mesh)
    signed_distance = (mesh.vertices - plane_origin) @ span_axis
    rough_thickness_axis = section_basis[:, 1]
    intersections: list[np.ndarray] = []

    for edge, faces in zip(edges, adjacent_faces):
        if len(faces) != 2:
            continue
        start, end = (int(value) for value in edge)
        first_distance = float(signed_distance[start])
        second_distance = float(signed_distance[end])
        if first_distance * second_distance > 0.0:
            continue
        denominator = first_distance - second_distance
        if abs(denominator) <= 1e-14:
            continue
        normal_product = float(
            np.dot(normals[faces[0]], rough_thickness_axis)
            * np.dot(normals[faces[1]], rough_thickness_axis)
        )
        normal_angle = float(
            np.arccos(
                np.clip(np.dot(normals[faces[0]], normals[faces[1]]), -1.0, 1.0)
            )
        )
        # A closure joins upper and lower surfaces. Their thickness-normal
        # components oppose one another, unlike edges within either surface.
        if normal_product >= -0.15 or normal_angle < np.deg2rad(90.0):
            continue
        fraction = first_distance / denominator
        intersections.append(
            mesh.vertices[start]
            + fraction * (mesh.vertices[end] - mesh.vertices[start])
        )

    if len(intersections) < 2:
        return None
    points = np.asarray(intersections, dtype=float)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    chord_axis = points[second] - points[first]
    chord_axis -= span_axis * float(np.dot(chord_axis, span_axis))
    norm = float(np.linalg.norm(chord_axis))
    return chord_axis / norm if norm > 1e-12 else None


def _surface_normal_chord_axis(mesh: TriMesh, span_axis: np.ndarray) -> np.ndarray | None:
    """Estimate chord from the dominant upper/lower surface-normal direction."""
    triangles = mesh.vertices[mesh.faces]
    area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_areas = np.linalg.norm(area_vectors, axis=1)
    valid = double_areas > 1e-15
    if not np.any(valid):
        return None
    normals = area_vectors[valid] / double_areas[valid, None]
    normal_second_moment = np.einsum(
        "f,fi,fj->ij",
        double_areas[valid],
        normals,
        normals,
    ) / float(np.sum(double_areas[valid]))
    projection = np.eye(3) - np.outer(span_axis, span_axis)
    normal_second_moment = projection @ normal_second_moment @ projection
    thickness_axis = np.linalg.eigh(normal_second_moment)[1][:, -1]
    thickness_axis -= span_axis * float(np.dot(thickness_axis, span_axis))
    thickness_norm = float(np.linalg.norm(thickness_axis))
    if thickness_norm <= 1e-12:
        return None
    thickness_axis /= thickness_norm
    chord_axis = np.cross(span_axis, thickness_axis)
    chord_norm = float(np.linalg.norm(chord_axis))
    return chord_axis / chord_norm if chord_norm > 1e-12 else None


def canonicalize_aircraft_mesh(mesh: TriMesh) -> tuple[TriMesh, CanonicalFrame]:
    """Align a wing-like surface using area moments and section geometry."""
    surface_centroid, covariance = _surface_area_moments(mesh)
    _, eigenvectors = np.linalg.eigh(covariance)
    candidates = eigenvectors[:, ::-1]
    projected = (mesh.vertices - surface_centroid) @ candidates
    extents = np.ptp(projected, axis=0)
    order = np.argsort(extents)[::-1]
    span_axis = candidates[:, order[0]]
    section_u = candidates[:, order[1]]
    section_v = candidates[:, order[2]]
    span_coordinate = (mesh.vertices - surface_centroid) @ span_axis
    root_offset = 0.5 * (float(span_coordinate.min()) + float(span_coordinate.max()))
    root_plane_origin = surface_centroid + root_offset * span_axis
    section_basis = np.column_stack([section_u, section_v])
    normal_chord_axis = _surface_normal_chord_axis(mesh, span_axis)
    closure_chord_axis = _closure_chord_axis(
        mesh,
        root_plane_origin,
        span_axis,
        section_basis,
    )
    chord_axis = normal_chord_axis
    if closure_chord_axis is not None and (
        chord_axis is None
        or abs(float(np.dot(closure_chord_axis, chord_axis))) >= np.cos(np.deg2rad(2.0))
    ):
        chord_axis = closure_chord_axis
    if chord_axis is None:
        root = np.abs(span_coordinate - root_offset) <= 0.10 * max(
            0.5 * np.ptp(span_coordinate), 1e-12
        )
        root_section = (mesh.vertices[root] - root_plane_origin) @ section_basis
        if len(root_section) >= 8:
            _, section_vectors = np.linalg.eigh(np.cov(root_section.T))
            chord_axis = section_basis @ section_vectors[:, -1]
        else:
            chord_axis = section_u
    chord_axis /= max(np.linalg.norm(chord_axis), 1e-12)
    thickness_axis = np.cross(chord_axis, span_axis)
    thickness_axis /= max(np.linalg.norm(thickness_axis), 1e-12)
    chord_axis = np.cross(span_axis, thickness_axis)
    chord_axis /= max(np.linalg.norm(chord_axis), 1e-12)
    axes = np.column_stack([chord_axis, span_axis, thickness_axis])
    projected = (mesh.vertices - root_plane_origin) @ axes

    # PCA leaves a fore/aft ambiguity. Resolve it from a root airfoil section:
    # conventional airfoils reach maximum thickness forward of mid-chord.
    # Closure angle and planform sweep are fallbacks for sparse STL sections.
    x = projected[:, 0]
    y = projected[:, 1]
    temporary = TriMesh(projected, mesh.faces)
    temporary_edges, adjacent_faces = edge_topology(temporary)
    normals = face_normals(temporary)
    starts = projected[temporary_edges[:, 0]]
    ends = projected[temporary_edges[:, 1]]
    vectors = ends - starts
    directions = vectors / np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-12)
    midpoints = 0.5 * (starts + ends)
    mid_x = midpoints[:, 0]
    spanwise = np.abs(directions[:, 1]) > 0.65
    dihedral = np.zeros(len(temporary_edges), dtype=float)
    for index, faces in enumerate(adjacent_faces):
        if len(faces) >= 2:
            cosine = float(np.clip(np.dot(normals[faces[0]], normals[faces[1]]), -1.0, 1.0))
            dihedral[index] = float(np.arccos(cosine))
    local_chord_fraction, _ = planform_coordinates(projected, midpoints)
    low_edges = spanwise & (local_chord_fraction <= 0.08)
    high_edges = spanwise & (local_chord_fraction >= 0.92)
    low_closure = float(np.mean(dihedral[low_edges])) if np.any(low_edges) else 0.0
    high_closure = float(np.mean(dihedral[high_edges])) if np.any(high_edges) else 0.0
    low_sharpness = float(np.quantile(dihedral[low_edges], 0.98)) if np.any(low_edges) else 0.0
    high_sharpness = float(np.quantile(dihedral[high_edges], 0.98)) if np.any(high_edges) else 0.0

    half_span = max(0.5 * np.ptp(y), 1e-12)
    thickness_samples: list[list[float]] = [[] for _ in range(20)]
    local_span_bins = np.linspace(y.min(), y.max(), 25)
    for span_low, span_high in zip(local_span_bins[:-1], local_span_bins[1:]):
        local = (y >= span_low) & (y <= span_high)
        local_x = x[local]
        local_z = projected[local, 2]
        if len(local_x) < 8 or np.ptp(local_x) <= 1e-12:
            continue
        fraction = (local_x - local_x.min()) / np.ptp(local_x)
        for chord_index in range(20):
            selected = (fraction >= chord_index / 20.0) & (fraction <= (chord_index + 1) / 20.0)
            if np.sum(selected) >= 2:
                thickness_samples[chord_index].append(float(np.ptp(local_z[selected])))
    thickness_profile = np.asarray(
        [np.median(values) if values else np.nan for values in thickness_samples], dtype=float
    )
    thickness_peak = None
    if np.any(np.isfinite(thickness_profile)) and np.nanmax(thickness_profile) > 1e-8:
        thickness_peak = (int(np.nanargmax(thickness_profile)) + 0.5) / 20.0

    span_bins = np.linspace(y.min(), y.max(), 25)
    absolute_span = []
    lower_envelope = []
    upper_envelope = []
    for low, high in zip(span_bins[:-1], span_bins[1:]):
        values = x[(y >= low) & (y <= high)]
        if len(values) >= 4:
            absolute_span.append(abs(0.5 * (low + high)))
            lower_envelope.append(float(np.quantile(values, 0.02)))
            upper_envelope.append(float(np.quantile(values, 0.98)))
    if len(absolute_span) >= 4:
        design = np.column_stack([np.ones(len(absolute_span)), absolute_span])
        lower_sweep = float(np.linalg.lstsq(design, lower_envelope, rcond=None)[0][1])
        upper_sweep = float(np.linalg.lstsq(design, upper_envelope, rcond=None)[0][1])
    else:
        lower_sweep = 0.0
        upper_sweep = 0.0
    if abs(high_sharpness - low_sharpness) > np.deg2rad(7.0):
        # A conventional closed section has its sharper closure at the
        # trailing edge. This remains stable under irregular triangulation.
        flip = low_sharpness > high_sharpness
    elif thickness_peak is not None:
        flip = thickness_peak > 0.5
    elif abs(high_closure - low_closure) > np.deg2rad(1.0):
        flip = low_closure > high_closure
    else:
        flip = upper_sweep > lower_sweep
    if flip:
        axes[:, 0] *= -1.0
        axes[:, 2] *= -1.0

    # Resolve the remaining top/bottom ambiguity from root-section camber.
    # Positive conventional camber bows the mean line toward +z. Symmetric
    # sections leave the sign unchanged because it has no aerodynamic effect.
    oriented = (mesh.vertices - root_plane_origin) @ axes
    oriented_y = oriented[:, 1]
    oriented_x = oriented[:, 0]
    oriented_z = oriented[:, 2]
    root = np.abs(oriented_y) <= 0.10 * max(0.5 * np.ptp(oriented_y), 1e-12)
    root_x = oriented_x[root]
    root_z = oriented_z[root]
    if len(root_x) >= 12:
        chord_bins = np.linspace(root_x.min(), root_x.max(), 17)
        centers = []
        mean_line = []
        for low, high in zip(chord_bins[:-1], chord_bins[1:]):
            values = root_z[(root_x >= low) & (root_x <= high)]
            if len(values) >= 2:
                centers.append(0.5 * (low + high))
                mean_line.append(0.5 * (values.min() + values.max()))
        if len(centers) >= 5:
            centers_array = np.asarray(centers)
            mean_line_array = np.asarray(mean_line)
            endpoint_line = np.interp(
                centers_array,
                [centers_array[0], centers_array[-1]],
                [mean_line_array[0], mean_line_array[-1]],
            )
            interior = (centers_array > np.quantile(centers_array, 0.15)) & (
                centers_array < np.quantile(centers_array, 0.85)
            )
            if float(np.mean(mean_line_array[interior] - endpoint_line[interior])) < -1e-4:
                axes[:, 2] *= -1.0

    # Fore/aft and upper/lower are physically constrained; span sign is not.
    # Use that remaining freedom to preserve a right-handed canonical frame.
    if np.linalg.det(axes) < 0.0:
        axes[:, 1] *= -1.0

    aligned = (mesh.vertices - root_plane_origin) @ axes
    scale = float(max(np.ptp(aligned[:, 0]), 1e-12))
    aligned_center = 0.5 * (aligned.min(axis=0) + aligned.max(axis=0))
    origin = root_plane_origin + aligned_center @ axes.T
    frame = CanonicalFrame(origin=origin, axes=axes, scale=scale)
    canonical = apply_frame(mesh.vertices, frame)
    metadata = {**mesh.metadata, "canonicalized": True}
    if "source_curves" in metadata:
        source_curves = metadata["source_curves"]
        metadata["source_curves"] = {
            "leading_edge": apply_frame(np.asarray(source_curves["leading_edge"]), frame),
            "trailing_edge": apply_frame(np.asarray(source_curves["trailing_edge"]), frame),
            "tips": [apply_frame(np.asarray(curve), frame) for curve in source_curves["tips"]],
        }
    if "source_tip_boundary_options" in metadata:
        metadata["source_tip_boundary_options"] = [
            [apply_frame(np.asarray(curve), frame) for curve in options]
            for options in metadata["source_tip_boundary_options"]
        ]
    if "source_root_chord" in metadata:
        metadata["source_root_chord"] = float(metadata["source_root_chord"]) / frame.scale
    result = TriMesh(
        canonical,
        mesh.faces.copy(),
        None if mesh.labels is None else mesh.labels.copy(),
        metadata,
    )
    return result, frame


def face_normals(mesh: TriMesh) -> np.ndarray:
    triangles = mesh.vertices[mesh.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    return normals


def edge_topology(mesh: TriMesh) -> tuple[np.ndarray, list[list[int]]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(mesh.faces):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(start), int(end))))
            edge_faces.setdefault(edge, []).append(face_index)
    edges = np.asarray(sorted(edge_faces), dtype=int)
    adjacent = [edge_faces[tuple(edge)] for edge in edges]
    return edges, adjacent


def vertex_differential_descriptors(mesh: TriMesh) -> np.ndarray:
    """Compute one-ring normal and covariance descriptors."""
    normals = face_normals(mesh)
    vertex_normals = np.zeros_like(mesh.vertices)
    incident_faces: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    neighbours: list[set[int]] = [set() for _ in range(len(mesh.vertices))]
    for face_index, face in enumerate(mesh.faces):
        a, b, c = (int(value) for value in face)
        for vertex in (a, b, c):
            vertex_normals[vertex] += normals[face_index]
            incident_faces[vertex].append(face_index)
        neighbours[a].update((b, c))
        neighbours[b].update((a, c))
        neighbours[c].update((a, b))

    lengths = np.linalg.norm(vertex_normals, axis=1)
    valid = lengths > 1e-12
    vertex_normals[valid] /= lengths[valid, None]
    descriptors = np.zeros((len(mesh.vertices), 7), dtype=float)
    descriptors[:, :3] = np.abs(vertex_normals)
    for index, face_indices in enumerate(incident_faces):
        if len(face_indices) >= 2:
            local_normals = normals[face_indices]
            mean_normal = np.mean(local_normals, axis=0)
            descriptors[index, 3] = float(
                np.sqrt(np.mean(np.sum((local_normals - mean_normal) ** 2, axis=1)))
            )
        local_neighbours = sorted(neighbours[index])
        if len(local_neighbours) < 3:
            continue
        offsets = mesh.vertices[local_neighbours] - mesh.vertices[index]
        covariance = offsets.T @ offsets / len(offsets)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        total = float(np.sum(eigenvalues))
        maximum = float(eigenvalues[-1])
        if total > 1e-15:
            descriptors[index, 4] = float(eigenvalues[0] / total)
        if maximum > 1e-15:
            descriptors[index, 5] = float((eigenvalues[2] - eigenvalues[1]) / maximum)
            descriptors[index, 6] = float((eigenvalues[1] - eigenvalues[0]) / maximum)
    return descriptors


def planform_coordinates(vertices: np.ndarray, points: np.ndarray, n_bins: int = 41) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local chord and absolute-span fractions from an unstructured surface."""
    span_min = float(vertices[:, 1].min())
    span_max = float(vertices[:, 1].max())
    bins = np.linspace(span_min, span_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    vertex_bins = np.clip(np.digitize(vertices[:, 1], bins) - 1, 0, n_bins - 1)
    lower = np.full(n_bins, np.nan, dtype=float)
    upper = np.full(n_bins, np.nan, dtype=float)
    for index in range(n_bins):
        local_x = vertices[vertex_bins == index, 0]
        if len(local_x):
            lower[index] = float(np.quantile(local_x, 0.01))
            upper[index] = float(np.quantile(local_x, 0.99))
    valid = np.isfinite(lower) & np.isfinite(upper)
    if np.sum(valid) < 2:
        lower[:] = float(vertices[:, 0].min())
        upper[:] = float(vertices[:, 0].max())
    else:
        lower = np.interp(centers, centers[valid], lower[valid])
        upper = np.interp(centers, centers[valid], upper[valid])
    local_lower = np.interp(points[:, 1], centers, lower)
    local_upper = np.interp(points[:, 1], centers, upper)
    chord_fraction = (points[:, 0] - local_lower) / np.maximum(local_upper - local_lower, 1e-12)
    center = 0.5 * (span_min + span_max)
    half_span = max(0.5 * (span_max - span_min), 1e-12)
    absolute_span = np.abs(points[:, 1] - center) / half_span
    return chord_fraction, absolute_span


def edge_feature_table(mesh: TriMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return edges, classical geometric descriptors, and optional labels."""
    edges, adjacent = edge_topology(mesh)
    normals = face_normals(mesh)
    starts = mesh.vertices[edges[:, 0]]
    ends = mesh.vertices[edges[:, 1]]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.maximum(lengths[:, None], 1e-12)
    midpoints = 0.5 * (starts + ends)

    minimum = mesh.vertices.min(axis=0)
    extent = np.maximum(np.ptp(mesh.vertices, axis=0), 1e-12)
    normalized_midpoints = (midpoints - minimum) / extent
    local_chord_fraction, absolute_span_fraction = planform_coordinates(mesh.vertices, midpoints)
    vertex_descriptors = vertex_differential_descriptors(mesh)
    edge_descriptors = 0.5 * (
        vertex_descriptors[edges[:, 0]] + vertex_descriptors[edges[:, 1]]
    )

    dihedral = np.zeros(len(edges), dtype=float)
    boundary = np.zeros(len(edges), dtype=float)
    for index, faces in enumerate(adjacent):
        if len(faces) == 1:
            boundary[index] = 1.0
            dihedral[index] = np.pi
        elif len(faces) >= 2:
            cosine = float(np.clip(np.dot(normals[faces[0]], normals[faces[1]]), -1.0, 1.0))
            dihedral[index] = float(np.arccos(cosine))

    features = np.column_stack(
        [
            normalized_midpoints,
            np.abs(directions),
            lengths / max(float(np.mean(lengths)), 1e-12),
            dihedral / np.pi,
            boundary,
            local_chord_fraction,
            absolute_span_fraction,
            np.abs(directions[:, 1]),
            edge_descriptors,
        ]
    )

    labels = None
    source_edge_labels = mesh.metadata.get("source_edge_labels")
    if source_edge_labels is not None:
        labels = np.asarray(
            [source_edge_labels.get(tuple(int(value) for value in edge), 0) for edge in edges],
            dtype=int,
        )
    elif mesh.labels is not None:
        labels = np.zeros(len(edges), dtype=int)
        start_labels = mesh.labels[edges[:, 0]]
        end_labels = mesh.labels[edges[:, 1]]
        same_feature = (start_labels == end_labels) & (start_labels > 0)
        labels[same_feature] = start_labels[same_feature]
    return edges, features, labels


def topology_metrics(mesh: TriMesh) -> dict[str, float]:
    edges, adjacent = edge_topology(mesh)
    edge_uses = _face_edge_uses(mesh.faces)
    triangles = mesh.vertices[mesh.faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    return {
        "vertices": float(len(mesh.vertices)),
        "faces": float(len(mesh.faces)),
        "edges": float(len(edges)),
        "boundary_edge_fraction": float(np.mean([len(faces) == 1 for faces in adjacent])),
        "nonmanifold_edge_fraction": float(np.mean([len(faces) > 2 for faces in adjacent])),
        "degenerate_face_fraction": float(np.mean(areas < 1e-12)),
        "boundary_edges": float(sum(len(uses) == 1 for uses in edge_uses.values())),
        "nonmanifold_edges": float(sum(len(uses) > 2 for uses in edge_uses.values())),
        "closed": float(bool(edge_uses) and all(len(uses) == 2 for uses in edge_uses.values())),
        "consistently_oriented": float(
            all(len(uses) != 2 or uses[0][1] != uses[1][1] for uses in edge_uses.values())
        ),
    }
