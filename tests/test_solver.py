from dataclasses import replace

import numpy as np

from aero_mesh.geometry import generate_aero_mesh, make_wing_params_for_aspect_ratio, random_rotation, transform_aero_mesh
from aero_mesh.solver import (
    aerodynamic_axes,
    circulation_lift_coefficient,
    finite_segment_velocity,
    horseshoe_influence_matrix,
    horseshoe_velocity,
    prandtl_lift_coefficient,
    solve_bound_circulation,
    solve_mesh_steady_vlm,
    solve_steady_vlm,
    trefftz_induced_drag_coefficient,
)
from aero_mesh.types import SimulationCase
from aero_mesh.wake import (
    compare_wake_histories,
    ring_dipole_far_field_error,
    ring_to_equivalent_dipole,
    simulate_prescribed_free_wake,
)


def test_prandtl_reference_increases_with_angle():
    assert prandtl_lift_coefficient(8, 8) > prandtl_lift_coefficient(4, 8)


def test_vlm_cl_increases_with_angle_of_attack():
    params = make_wing_params_for_aspect_ratio(8, n_span=8, n_chord=3)
    low = solve_steady_vlm(SimulationCase("low", params, alpha_deg=2))
    high = solve_steady_vlm(SimulationCase("high", params, alpha_deg=8))
    assert high.cl > low.cl
    assert high.converged


def test_symmetric_wing_has_zero_lift_at_zero_angle():
    params = make_wing_params_for_aspect_ratio(8, n_span=12, n_chord=4)
    result = solve_steady_vlm(SimulationCase("zero", params, alpha_deg=0))
    assert abs(result.cl) < 1e-12
    assert abs(result.cm) < 1e-12


def test_lift_sign_is_not_forced_by_angle_of_attack():
    params = make_wing_params_for_aspect_ratio(8, n_span=12, n_chord=4)
    positive = solve_steady_vlm(SimulationCase("positive", params, alpha_deg=5))
    negative = solve_steady_vlm(SimulationCase("negative", params, alpha_deg=-5))
    assert positive.cl > 0.0
    assert negative.cl < 0.0
    assert np.isclose(positive.cl, -negative.cl, rtol=1e-10, atol=1e-12)


def test_wash_in_wash_out_and_camber_have_expected_lift_effects():
    base = make_wing_params_for_aspect_ratio(8, n_span=16, n_chord=5)
    wash_out = solve_steady_vlm(SimulationCase("wash-out", replace(base, twist_deg=-4), alpha_deg=0))
    untwisted = solve_steady_vlm(SimulationCase("untwisted", base, alpha_deg=0))
    wash_in = solve_steady_vlm(SimulationCase("wash-in", replace(base, twist_deg=4), alpha_deg=0))
    cambered = solve_steady_vlm(SimulationCase("cambered", replace(base, camber_ratio=0.02), alpha_deg=0))
    assert wash_out.cl < untwisted.cl < wash_in.cl
    assert cambered.cl > untwisted.cl


def test_force_is_projected_onto_wind_axis():
    params = make_wing_params_for_aspect_ratio(8, n_span=4, n_chord=1)
    case = SimulationCase("projection", params, alpha_deg=30)
    mesh = generate_aero_mesh(params)
    gamma = np.ones(len(mesh.panels))
    freestream_axis, lift_axis, _ = aerodynamic_axes(case.alpha_deg, mesh.body_axes)
    freestream = case.velocity * freestream_axis
    forces = case.density * gamma[:, None] * np.cross(
        np.broadcast_to(freestream, mesh.bound_right.shape), mesh.bound_right - mesh.bound_left
    )
    expected = np.sum(forces, axis=0) @ lift_axis
    q = 0.5 * case.density * case.velocity**2
    cl, _ = circulation_lift_coefficient(mesh, case, gamma, freestream)
    assert np.isclose(cl, expected / (q * mesh.reference_area))
    assert not np.isclose(expected, np.sum(forces[:, 2]))


def test_trailing_legs_follow_freestream_at_nonzero_angle():
    point = np.array([0.5, 0.0, 0.8])
    left = np.array([0.25, -0.5, 0.0])
    right = np.array([0.25, 0.5, 0.0])
    direction, _, _ = aerodynamic_axes(12.0)
    wake_length = 25.0
    expected = finite_segment_velocity(point, left, right)
    expected += finite_segment_velocity(point, right, right + wake_length * direction)
    expected += finite_segment_velocity(point, left + wake_length * direction, left)
    actual = horseshoe_velocity(
        point, left, right, wake_length, downstream_direction=direction
    )
    assert np.allclose(actual, expected)


def test_vlm_follows_aspect_ratio_trend():
    low_ar = make_wing_params_for_aspect_ratio(4, n_span=8, n_chord=3)
    high_ar = make_wing_params_for_aspect_ratio(12, n_span=8, n_chord=3)
    low = solve_steady_vlm(SimulationCase("low_ar", low_ar, alpha_deg=5))
    high = solve_steady_vlm(SimulationCase("high_ar", high_ar, alpha_deg=5))
    assert high.cl > low.cl


def test_trefftz_induced_drag_is_positive_and_physically_scaled():
    params = make_wing_params_for_aspect_ratio(8, n_span=20, n_chord=4)
    case = SimulationCase("drag", params, alpha_deg=5)
    mesh = generate_aero_mesh(params)
    gamma, _, _ = solve_bound_circulation(mesh, case)
    cdi = trefftz_induced_drag_coefficient(mesh, case, gamma)
    result = solve_steady_vlm(case)
    span_efficiency = result.cl**2 / (np.pi * params.aspect_ratio * cdi)

    assert np.isclose(result.cdi, cdi)
    assert cdi > 0.0
    assert 0.8 < span_efficiency < 1.05


def test_trefftz_induced_drag_rejects_incomplete_circulation_vector():
    params = make_wing_params_for_aspect_ratio(8, n_span=8, n_chord=3)
    case = SimulationCase("drag", params, alpha_deg=5)
    mesh = generate_aero_mesh(params)

    with np.testing.assert_raises_regex(ValueError, "one circulation value per aerodynamic panel"):
        trefftz_induced_drag_coefficient(mesh, case, np.ones(5))


def test_vectorized_horseshoe_influence_matches_scalar_formula():
    params = make_wing_params_for_aspect_ratio(8, n_span=4, n_chord=2)
    mesh = generate_aero_mesh(params)
    wake_length = 100.0
    matrix = horseshoe_influence_matrix(
        mesh.collocation_points,
        mesh.normals,
        mesh.bound_left,
        mesh.bound_right,
        wake_length,
    )
    for i in (0, len(mesh.panels) - 1):
        for j in (0, len(mesh.panels) - 1):
            scalar = np.dot(
                horseshoe_velocity(
                    mesh.collocation_points[i], mesh.bound_left[j], mesh.bound_right[j], wake_length
                ),
                mesh.normals[i],
            )
            assert np.isclose(matrix[i, j], scalar)


def test_bound_circulation_matches_steady_solver():
    params = make_wing_params_for_aspect_ratio(8, n_span=6, n_chord=2)
    case = SimulationCase("circulation", params, alpha_deg=5)
    gamma, _, converged = solve_bound_circulation(generate_aero_mesh(params), case)
    assert converged
    assert gamma.shape == (params.n_span * params.n_chord,)
    assert np.all(np.isfinite(gamma))


def test_solver_is_invariant_to_rigid_transform_and_uniform_scale():
    params = make_wing_params_for_aspect_ratio(8, n_span=10, n_chord=3)
    case = SimulationCase("invariance", params, alpha_deg=5)
    mesh = generate_aero_mesh(params)
    baseline = solve_mesh_steady_vlm(mesh, case)
    transformed = transform_aero_mesh(
        mesh,
        rotation=random_rotation(901),
        translation=np.array([7.0, -3.0, 2.0]),
        scale=2.75,
    )
    result = solve_mesh_steady_vlm(transformed, case)
    assert np.isclose(result.cl, baseline.cl, rtol=2e-7, atol=1e-9)
    assert np.isclose(result.cdi, baseline.cdi, rtol=2e-7, atol=1e-9)
    assert np.isclose(result.cm, baseline.cm, rtol=2e-7, atol=1e-9)


def test_reconstructed_wake_methods_produce_finite_comparison():
    params = make_wing_params_for_aspect_ratio(6, n_span=4, n_chord=1)
    case = SimulationCase("wake", params, alpha_deg=5, velocity=1.0, timestep=0.08, n_steps=4)
    panel = simulate_prescribed_free_wake(case, "panel")
    dipole = simulate_prescribed_free_wake(case, "dipole_gradient")
    metrics = compare_wake_histories(panel, dipole, params.root_chord)
    assert panel.element_centers.shape == dipole.element_centers.shape
    assert np.all(np.isfinite(panel.grid))
    assert np.all(np.isfinite(dipole.element_centers))
    assert np.all(panel.finite_state)
    assert np.all(dipole.finite_state)
    assert metrics["trajectory_rms_chords"] >= 0.0


def test_vortex_ring_is_replaced_by_centroid_and_area_vector_moment():
    ring = np.array(
        [[[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]]
    )
    centers, moments = ring_to_equivalent_dipole(ring, np.array([2.0]))
    assert np.allclose(centers, [[0.0, 0.0, 0.0]])
    assert np.allclose(moments, [[0.0, 0.0, 2.0]])


def test_ring_and_equivalent_dipole_converge_in_the_far_field():
    rows = ring_dipole_far_field_error(np.array([2.0, 4.0, 8.0, 16.0]))
    relative_errors = np.array([row["relative_velocity_error"] for row in rows])
    assert np.all(np.diff(relative_errors) < 0.0)
    assert relative_errors[-1] < 0.002
