import numpy as np

from aero_mesh.geometry import (
    LABEL_LEADING_EDGE,
    LABEL_TRAILING_EDGE,
    aerodynamic_reference,
    extract_geometry_features,
    generate_aero_mesh,
    generate_wing_surface,
    make_wing_params_for_aspect_ratio,
    panel_quality,
    rule_based_labels,
)
from aero_mesh.dataset import generate_geometry_cases


def test_parametric_surface_has_expected_feature_labels():
    params = make_wing_params_for_aspect_ratio(8, n_span=6, n_chord=3)
    vertices, faces, labels = generate_wing_surface(params)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert np.sum(labels == LABEL_LEADING_EDGE) > 0
    assert np.sum(labels == LABEL_TRAILING_EDGE) > 0


def test_rule_based_labels_have_correct_length():
    params = make_wing_params_for_aspect_ratio(8, sweep_deg=15, taper_ratio=0.6)
    vertices, faces, labels = generate_wing_surface(params, noise=0.001)
    features = extract_geometry_features(vertices, faces, labels)
    predicted = rule_based_labels(features)
    assert predicted.shape == labels.shape


def test_generated_aero_mesh_is_valid():
    params = make_wing_params_for_aspect_ratio(8, n_span=8, n_chord=3)
    mesh = generate_aero_mesh(params)
    quality = panel_quality(mesh)
    assert len(mesh.panels) == 24
    assert quality["valid_panel_fraction"] == 1.0
    assert quality["min_area"] > 0.0


def test_trapezoidal_reference_quantities_match_analytic_values():
    params = make_wing_params_for_aspect_ratio(8, taper_ratio=0.4)
    area, mean_chord, span, reference = aerodynamic_reference(params)
    expected_mean_chord = (
        2.0
        * params.root_chord
        * (1.0 + params.taper_ratio + params.taper_ratio**2)
        / (3.0 * (1.0 + params.taper_ratio))
    )
    assert np.isclose(area, params.area, rtol=1e-7)
    assert np.isclose(mean_chord, expected_mean_chord, rtol=1e-7)
    assert np.isclose(span, params.span)
    assert np.isclose(reference[0], 0.25 * expected_mean_chord, rtol=1e-7)


def test_requested_aspect_ratio_is_preserved_for_all_planforms():
    for family in ("trapezoidal", "elliptical", "cranked"):
        params = make_wing_params_for_aspect_ratio(
            11.5,
            taper_ratio=0.43,
            planform_family=family,
            kink_span_fraction=0.37,
            kink_chord_ratio=0.68,
        )
        assert np.isclose(params.aspect_ratio, 11.5, rtol=2e-6)


def test_generated_geometry_identifiers_are_unique_across_dataset_seeds():
    first = generate_geometry_cases({"trapezoidal": 1}, seed=13)
    second = generate_geometry_cases({"trapezoidal": 1}, seed=29)
    assert first[0].base_geometry_id != second[0].base_geometry_id
    assert first[0].mesh_id != second[0].mesh_id
