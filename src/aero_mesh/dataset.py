from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .geometry import MESHERS, generate_remeshed_wing_trimesh, make_wing_params_for_aspect_ratio
from .stl import canonicalize_aircraft_mesh, edge_feature_table
from .types import TriMesh, WingParameters


SEEN_MESHERS = ("structured_alternating", "delaunay_uniform", "delaunay_random")
UNSEEN_MESHER = "delaunay_anisotropic"
MESH_DENSITY_LEVELS = ("coarse", "medium", "fine")


@dataclass
class GeometryCase:
    name: str
    family: str
    params: WingParameters
    mesh: TriMesh
    edges: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    base_geometry_id: str
    mesh_id: str
    mesher: str
    mesh_density: str
    remesh_seed: int
    source_kind: str


def rule_based_edge_labels(features: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(features), dtype=int)
    chord_fraction = features[:, 9]
    span_fraction = features[:, 10]
    dihedral = features[:, 7]
    boundary = features[:, 8]
    candidate = (dihedral > 0.04) | (boundary > 0.5)
    labels[candidate & (chord_fraction < 0.055)] = 1
    labels[candidate & (chord_fraction > 0.945)] = 2
    labels[candidate & (span_fraction > 0.965)] = 3
    return labels


def planform_envelope_edge_labels(features: np.ndarray, margin: float = 0.025) -> np.ndarray:
    """Select aerodynamic boundaries from local planform envelopes."""
    labels = np.zeros(len(features), dtype=int)
    chord_fraction = features[:, 9]
    span_fraction = features[:, 10]
    labels[chord_fraction < margin] = 1
    labels[chord_fraction > 1.0 - margin] = 2
    labels[span_fraction > 0.99] = 3
    return labels


def _base_parameters(index: int, family: str, seed: int) -> WingParameters:
    rng = np.random.default_rng(seed + index * 97)
    aspect_ratio = float(rng.uniform(4.0, 18.0))
    sweep = float(rng.uniform(-4.0, 28.0))
    taper = float(rng.uniform(0.28, 0.95))
    kink_span_fraction = float(rng.uniform(0.30, 0.62))
    kink_chord_ratio = float(rng.uniform(0.50, 0.88))
    params = make_wing_params_for_aspect_ratio(
        aspect_ratio,
        name=f"{family}_s{seed}_{index:04d}",
        sweep_deg=sweep,
        taper_ratio=taper,
        n_span=int(rng.integers(8, 14)),
        n_chord=int(rng.integers(3, 6)),
        planform_family=family,
        kink_span_fraction=kink_span_fraction,
        kink_chord_ratio=kink_chord_ratio,
    )
    return replace(
        params,
        dihedral_deg=float(rng.uniform(-3.0, 9.0)),
        twist_deg=float(rng.uniform(-5.0, 4.0)),
        camber_ratio=float(rng.uniform(0.0, 0.045)),
        thickness_ratio=float(rng.uniform(0.08, 0.18)),
    )


def _make_case(
    index: int,
    family: str,
    seed: int,
    mesher: str,
    mesh_density: str,
    mesher_index: int,
) -> GeometryCase:
    params = _base_parameters(index, family, seed)
    base_geometry_id = params.name
    remesh_seed = seed * 1_000_003 + index * 101 + mesher_index * 7_919
    mesh_id = f"{base_geometry_id}__{mesher}__{mesh_density}"
    raw = generate_remeshed_wing_trimesh(
        params,
        mesher=mesher,
        mesh_density=mesh_density,
        seed=remesh_seed,
        rotate=True,
    )
    raw.metadata.update({"base_geometry_id": base_geometry_id, "mesh_id": mesh_id})
    canonical, _ = canonicalize_aircraft_mesh(raw)
    edges, features, labels = edge_feature_table(canonical)
    if labels is None:
        raise RuntimeError("Synthetic case lost source-curve edge labels")
    return GeometryCase(
        name=mesh_id,
        family=family,
        params=params,
        mesh=canonical,
        edges=edges,
        features=features,
        labels=labels,
        base_geometry_id=base_geometry_id,
        mesh_id=mesh_id,
        mesher=mesher,
        mesh_density=mesh_density,
        remesh_seed=remesh_seed,
        source_kind="synthetic_parametric",
    )


def generate_geometry_cases(
    counts: dict[str, int] | None = None,
    seed: int = 13,
    meshers: tuple[str, ...] | list[str] | None = None,
) -> list[GeometryCase]:
    """Generate four independent triangulations of each physical geometry."""
    requested = counts or {"trapezoidal": 50, "elliptical": 30, "cranked": 30}
    requested_meshers = tuple(MESHERS if meshers is None else meshers)
    unknown = set(requested_meshers) - set(MESHERS)
    if unknown:
        raise ValueError(f"unknown meshers: {sorted(unknown)}")
    cases: list[GeometryCase] = []
    index = 0
    for family, count in requested.items():
        for _ in range(count):
            density = MESH_DENSITY_LEVELS[index % len(MESH_DENSITY_LEVELS)]
            for mesher_index, mesher in enumerate(requested_meshers):
                cases.append(_make_case(index, family, seed, mesher, density, mesher_index))
            index += 1
    return cases


def base_geometry_ids(cases: list[GeometryCase], family: str) -> list[str]:
    return list(dict.fromkeys(case.base_geometry_id for case in cases if case.family == family))


def cases_for_base_ids(
    cases: list[GeometryCase],
    identifiers: set[str] | list[str] | tuple[str, ...],
    meshers: tuple[str, ...] | list[str] | None = None,
) -> list[GeometryCase]:
    selected = set(identifiers)
    allowed_meshers = None if meshers is None else set(meshers)
    return [
        case
        for case in cases
        if case.base_geometry_id in selected
        and (allowed_meshers is None or case.mesher in allowed_meshers)
    ]


def split_geometry_cases(cases: list[GeometryCase]) -> dict[str, list[GeometryCase]]:
    """Create leakage-free base-geometry splits and an unseen-mesher test."""
    trapezoidal = base_geometry_ids(cases, "trapezoidal")
    elliptical = base_geometry_ids(cases, "elliptical")
    cranked = base_geometry_ids(cases, "cranked")
    train_ids = set(trapezoidal[:35] + elliptical[:25])
    validation_ids = set(trapezoidal[35:45] + elliptical[25:33])
    iid_ids = set(trapezoidal[45:55] + elliptical[33:41])
    family_ids = set(cranked[:20])
    splits = {
        "train": cases_for_base_ids(cases, train_ids, SEEN_MESHERS),
        "validation": cases_for_base_ids(cases, validation_ids, SEEN_MESHERS),
        "iid": cases_for_base_ids(cases, iid_ids, SEEN_MESHERS),
        "unseen_mesher": cases_for_base_ids(cases, iid_ids, (UNSEEN_MESHER,)),
        "family_shift": cases_for_base_ids(cases, family_ids, SEEN_MESHERS),
        "family_shift_unseen_mesher": cases_for_base_ids(cases, family_ids, (UNSEEN_MESHER,)),
    }
    assert_split_integrity(splits)
    return splits


def assert_split_integrity(splits: dict[str, list[GeometryCase]]) -> None:
    fitting_ids = {
        case.base_geometry_id for name in ("train", "validation") for case in splits[name]
    }
    test_ids = {
        case.base_geometry_id
        for name in ("iid", "unseen_mesher", "family_shift", "family_shift_unseen_mesher")
        for case in splits[name]
    }
    overlap = fitting_ids & test_ids
    if overlap:
        raise ValueError(f"base geometries cross fitting and test splits: {sorted(overlap)}")
    fitting_meshers = {case.mesher for name in ("train", "validation") for case in splits[name]}
    if UNSEEN_MESHER in fitting_meshers:
        raise ValueError("the unseen mesher appears in a fitting split")
    if any(case.mesher != UNSEEN_MESHER for case in splits["unseen_mesher"]):
        raise ValueError("unseen-mesher split contains a seen triangulation procedure")


def balanced_case_arrays(
    cases: list[GeometryCase],
    seed: int = 13,
    surface_ratio: int = 2,
    maximum_per_boundary_class: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    feature_rows = []
    label_rows = []
    for case in cases:
        selected_features = []
        for label in (1, 2, 3):
            class_indices = np.flatnonzero(case.labels == label)
            if len(class_indices) > maximum_per_boundary_class:
                class_indices = rng.choice(
                    class_indices, size=maximum_per_boundary_class, replace=False
                )
            selected_features.append(class_indices)
        feature_indices = np.concatenate(selected_features)
        surface_indices = np.flatnonzero(case.labels == 0)
        maximum_surface = min(len(surface_indices), max(len(feature_indices) * surface_ratio, 1))
        if len(surface_indices) > maximum_surface:
            surface_indices = rng.choice(surface_indices, size=maximum_surface, replace=False)
        indices = np.concatenate([feature_indices, surface_indices])
        rng.shuffle(indices)
        feature_rows.append(case.features[indices])
        label_rows.append(case.labels[indices])
    return np.vstack(feature_rows), np.concatenate(label_rows)


def full_case_arrays(cases: list[GeometryCase]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    case_indices = []
    for index, case in enumerate(cases):
        features.append(case.features)
        labels.append(case.labels)
        case_indices.append(np.full(len(case.labels), index, dtype=int))
    return np.vstack(features), np.concatenate(labels), np.concatenate(case_indices)
