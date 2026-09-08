from pathlib import Path

import numpy as np

from aero_mesh.dataset import (
    SEEN_MESHERS,
    UNSEEN_MESHER,
    generate_geometry_cases,
    split_geometry_cases,
)
from aero_mesh.geometry import (
    MESHERS,
    generate_remeshed_wing_trimesh,
    generate_wing_trimesh,
    make_wing_params_for_aspect_ratio,
)
from aero_mesh.reconstruction import reconstruct_aero_mesh_from_edges, symmetric_curve_distance
from aero_mesh.stl import (
    canonicalize_aircraft_mesh,
    load_gmsh_v2_surface,
    load_stl,
    mirror_half_surface,
    repair_trimesh,
    topology_metrics,
    write_binary_stl,
)


def test_binary_stl_round_trip(tmp_path: Path):
    params = make_wing_params_for_aspect_ratio(8, n_span=8, n_chord=3)
    source = generate_wing_trimesh(params)
    path = write_binary_stl(tmp_path / "wing.stl", source)
    loaded = load_stl(path)
    assert loaded.faces.shape == source.faces.shape
    assert loaded.vertices.shape == source.vertices.shape
    assert np.allclose(np.sort(loaded.vertices, axis=0), np.sort(source.vertices, axis=0), atol=1e-6)


def test_load_gmsh_surface_and_mirror(tmp_path: Path):
    source = tmp_path / "half.msh"
    source.write_text(
        """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
2
1 2 2 7 1 1 2 3
2 2 2 7 1 1 3 4
$EndElements
""",
        encoding="utf-8",
    )
    half = load_gmsh_v2_surface(source, physical_tag=7)
    full = mirror_half_surface(half, axis=1)
    assert half.faces.shape == (2, 3)
    assert full.faces.shape == (4, 3)
    assert np.isclose(full.vertices[:, 1].min(), -1.0)
    assert np.isclose(full.vertices[:, 1].max(), 1.0)


def test_canonicalisation_resolves_leading_edge_direction():
    cases = generate_geometry_cases({"trapezoidal": 4, "elliptical": 4, "cranked": 4})
    for case in cases:
        leading_x = np.mean(case.mesh.vertices[case.mesh.labels == 1, 0])
        trailing_x = np.mean(case.mesh.vertices[case.mesh.labels == 2, 0])
        assert leading_x < trailing_x


def test_ground_truth_edges_reconstruct_all_planform_families():
    cases = generate_geometry_cases({"trapezoidal": 3, "elliptical": 3, "cranked": 3})
    for case in cases:
        result = reconstruct_aero_mesh_from_edges(case.mesh, case.edges, case.labels)
        assert result.valid, (case.name, result.diagnostics)
        assert result.mesh is not None
        assert len(result.mesh.panels) == 80


def test_canonical_frame_reproduces_returned_vertices():
    params = make_wing_params_for_aspect_ratio(10, sweep_deg=18, taper_ratio=0.45)
    source = generate_wing_trimesh(params, seed=9, rotate=True)
    canonical, frame = canonicalize_aircraft_mesh(source)
    expected = ((source.vertices - frame.origin) @ frame.axes) / frame.scale
    assert np.allclose(canonical.vertices, expected)


def test_canonical_planform_is_stable_across_independent_triangulations():
    params = make_wing_params_for_aspect_ratio(
        11.636874010631757,
        name="canonicalisation_regression",
        sweep_deg=12.122619166456769,
        taper_ratio=0.5403264632628838,
        n_span=9,
        n_chord=5,
        planform_family="elliptical",
    )
    params = type(params)(
        **{
            **params.__dict__,
            "dihedral_deg": 8.463541180495637,
            "twist_deg": -2.062652994494305,
            "thickness_ratio": 0.08592642425008931,
            "camber_ratio": 0.030806781987185238,
        }
    )
    measured = []
    for mesher_index, mesher in enumerate(MESHERS):
        raw = generate_remeshed_wing_trimesh(
            params,
            mesher=mesher,
            mesh_density="coarse",
            seed=47 * 1_000_003 + 63 * 101 + mesher_index * 7_919,
            rotate=True,
        )
        canonical, _ = canonicalize_aircraft_mesh(raw)
        curves = canonical.metadata["source_curves"]
        leading = np.asarray(curves["leading_edge"])
        trailing = np.asarray(curves["trailing_edge"])
        order = np.argsort(leading[:, 1])
        span_stations = leading[order, 1]
        projected_chord = np.abs(trailing[order, 0] - leading[order, 0])
        area = np.trapezoid(projected_chord, span_stations)
        measured.append(np.ptp(span_stations) ** 2 / area)

    assert np.max(np.abs(np.asarray(measured) - params.aspect_ratio)) < 0.02


def test_canonicalisation_preserves_positive_camber_direction():
    params = make_wing_params_for_aspect_ratio(8, n_span=10, n_chord=4)
    params = type(params)(**{**params.__dict__, "camber_ratio": 0.04})
    source = generate_wing_trimesh(params, seed=21, rotate=True)
    canonical, _ = canonicalize_aircraft_mesh(source)
    root = np.abs(canonical.vertices[:, 1]) < 0.05 * np.ptp(canonical.vertices[:, 1])
    points = canonical.vertices[root]
    center_x = 0.5 * (points[:, 0].min() + points[:, 0].max())
    middle = points[np.abs(points[:, 0] - center_x) < 0.1 * np.ptp(points[:, 0]), 2]
    edges = points[
        (points[:, 0] < points[:, 0].min() + 0.05 * np.ptp(points[:, 0]))
        | (points[:, 0] > points[:, 0].max() - 0.05 * np.ptp(points[:, 0])),
        2,
    ]
    assert np.mean(middle) > np.mean(edges)


def test_reconstruction_rejects_implausibly_narrow_chord():
    case = generate_geometry_cases({"trapezoidal": 1}, seed=91)[0]
    chord_fraction = case.features[:, 9]
    labels = np.zeros(len(case.edges), dtype=int)
    labels[(chord_fraction > 0.15) & (chord_fraction < 0.32)] = 1
    labels[(chord_fraction > 0.68) & (chord_fraction < 0.85)] = 2
    labels[case.labels == 3] = 3
    result = reconstruct_aero_mesh_from_edges(case.mesh, case.edges, labels)
    assert not result.valid
    assert result.failure_code == "IMPLAUSIBLE_CHORD"


def test_each_geometry_has_connectivity_distinct_remeshings_and_exact_labels():
    cases = generate_geometry_cases({"trapezoidal": 1}, seed=23)
    assert len(cases) == 4
    assert len({case.base_geometry_id for case in cases}) == 1
    assert len({case.mesher for case in cases}) == 4
    assert all(set(np.unique(case.labels)) == {0, 1, 2, 3} for case in cases)
    connectivity = {
        tuple(sorted(tuple(sorted(face)) for face in case.mesh.faces)) for case in cases
    }
    assert len(connectivity) > 1


def test_dataset_splits_are_grouped_by_geometry_and_hold_out_one_mesher():
    cases = generate_geometry_cases(
        {"trapezoidal": 55, "elliptical": 41, "cranked": 20}, seed=13
    )
    splits = split_geometry_cases(cases)
    fitting_ids = {
        case.base_geometry_id for split in ("train", "validation") for case in splits[split]
    }
    test_ids = {
        case.base_geometry_id
        for split in ("iid", "unseen_mesher", "family_shift")
        for case in splits[split]
    }
    assert fitting_ids.isdisjoint(test_ids)
    assert {case.mesher for case in splits["train"]} == set(SEEN_MESHERS)
    assert {case.mesher for case in splits["unseen_mesher"]} == {UNSEEN_MESHER}


def test_relative_welding_is_invariant_to_geometry_scale():
    params = make_wing_params_for_aspect_ratio(8, n_span=6, n_chord=3)
    source = generate_wing_trimesh(params)
    triangles = source.vertices[source.faces]
    expected = None
    for scale in (1e-6, 1.0, 1e6):
        vertices = (scale * triangles).reshape((-1, 3))
        faces = np.arange(len(vertices), dtype=int).reshape((-1, 3))
        repaired = repair_trimesh(type(source)(vertices, faces), intentionally_open=False)
        signature = (len(repaired.vertices), len(repaired.faces))
        expected = signature if expected is None else expected
        assert signature == expected
        assert topology_metrics(repaired)["closed"] == 1.0


def test_repair_removes_duplicate_and_zero_area_faces_and_orients_components():
    params = make_wing_params_for_aspect_ratio(8, n_span=6, n_chord=3)
    source = generate_wing_trimesh(params)
    faces = source.faces.copy()
    faces[::3] = faces[::3][:, [0, 2, 1]]
    corrupted = np.vstack([faces, faces[:5], np.array([[0, 0, 1]])])
    repaired = repair_trimesh(type(source)(source.vertices.copy(), corrupted), intentionally_open=False)
    metrics = topology_metrics(repaired)
    assert len(repaired.faces) == len(source.faces)
    assert repaired.metadata["removed_duplicate_faces"] == 5
    assert repaired.metadata["removed_collapsed_faces"] == 1
    assert metrics["closed"] == 1.0
    assert metrics["consistently_oriented"] == 1.0


def test_missing_facet_is_reported_but_not_mixed_with_accuracy_cases():
    params = make_wing_params_for_aspect_ratio(8, n_span=6, n_chord=3)
    source = generate_wing_trimesh(params)
    repaired = repair_trimesh(
        type(source)(source.vertices.copy(), source.faces[:-1].copy()), intentionally_open=False
    )
    metrics = topology_metrics(repaired)
    assert metrics["closed"] == 0.0
    assert metrics["boundary_edges"] > 0.0
    assert repaired.metadata["intentionally_open"] is False


def test_boundary_distance_measures_polylines_not_sampling_density():
    sparse = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    dense = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.25, 0.25, 0.0],
            [1.75, 0.75, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    assert symmetric_curve_distance(sparse, dense) < 1e-12


def test_boundary_distance_detects_curve_offset():
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    estimate = reference + np.array([0.0, 0.2, 0.0])
    assert np.isclose(symmetric_curve_distance(reference, estimate), 0.2)
