from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .geometry import generate_aero_mesh
from .solver import aerodynamic_axes, circulation_lift_coefficient
from .types import AeroMesh, SimulationCase


@dataclass
class WakeHistory:
    method: str
    element_centers: np.ndarray
    moments: np.ndarray
    cl: np.ndarray
    downwash_rms: np.ndarray
    centroid: np.ndarray
    moment_mean: np.ndarray
    minimum_area: np.ndarray
    finite_state: np.ndarray
    runtime_s: float
    grid: np.ndarray | None = None


def _finite_segment_velocity_core(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    strengths: np.ndarray,
    core_radius: float,
) -> np.ndarray:
    """Regularized Biot-Savart velocity from many finite segments."""
    if len(starts) == 0:
        return np.zeros_like(points)
    r1 = points[:, None, :] - starts[None, :, :]
    r2 = points[:, None, :] - ends[None, :, :]
    r0 = ends - starts
    cross = np.cross(r1, r2)
    cross_norm_sq = np.einsum("ijk,ijk->ij", cross, cross)
    r1_norm = np.linalg.norm(r1, axis=2)
    r2_norm = np.linalg.norm(r2, axis=2)
    direction = r1 / np.maximum(r1_norm[:, :, None], 1e-12)
    direction -= r2 / np.maximum(r2_norm[:, :, None], 1e-12)
    projection = np.einsum("jk,ijk->ij", r0, direction)
    segment_length_sq = np.einsum("ij,ij->i", r0, r0)
    denominator = 4.0 * math.pi * (
        cross_norm_sq + core_radius**2 * segment_length_sq[None, :]
    )
    factor = strengths[None, :] * projection / np.maximum(denominator, 1e-14)
    valid = (r1_norm >= 1e-10) & (r2_norm >= 1e-10)
    factor[~valid] = 0.0
    return np.sum(factor[:, :, None] * cross, axis=1)


def _ring_area_vectors(rings: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(rings[:, 0], rings[:, 1])
        + np.cross(rings[:, 1], rings[:, 2])
        + np.cross(rings[:, 2], rings[:, 3])
        + np.cross(rings[:, 3], rings[:, 0])
    )


def ring_to_equivalent_dipole(
    rings: np.ndarray,
    strengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the centroid and circulation-weighted area vector of each ring."""
    rings = np.asarray(rings, dtype=float).reshape((-1, 4, 3))
    strengths = np.asarray(strengths, dtype=float).reshape(-1)
    if len(rings) != len(strengths):
        raise ValueError("rings and strengths must contain the same number of elements")
    return rings.mean(axis=1), strengths[:, None] * _ring_area_vectors(rings)


def ring_dipole_far_field_error(
    distances: np.ndarray,
    core_radius: float = 1e-6,
) -> list[dict[str, float]]:
    """Compare a unit square vortex ring with its equivalent point dipole."""
    ring = np.asarray(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        dtype=float,
    )[None, :, :]
    strengths = np.ones(1, dtype=float)
    centers, moments = ring_to_equivalent_dipole(ring, strengths)
    rows: list[dict[str, float]] = []
    for distance in np.asarray(distances, dtype=float):
        if distance <= 0.0:
            raise ValueError("far-field distances must be positive")
        direction = np.asarray([1.0, 0.37, 0.21], dtype=float)
        direction /= np.linalg.norm(direction)
        point = centers + distance * direction[None, :]
        ring_velocity = _ring_element_velocity(point, ring, strengths, core_radius)[0]
        dipole_velocity = _dipole_element_velocity(point, centers, moments, core_radius)[0]
        absolute_error = float(np.linalg.norm(ring_velocity - dipole_velocity))
        reference = max(float(np.linalg.norm(ring_velocity)), 1e-15)
        rows.append(
            {
                "distance_over_ring_length": float(distance),
                "ring_velocity_norm": float(np.linalg.norm(ring_velocity)),
                "dipole_velocity_norm": float(np.linalg.norm(dipole_velocity)),
                "absolute_velocity_error": absolute_error,
                "relative_velocity_error": absolute_error / reference,
            }
        )
    return rows


def _grid_rings(grid: np.ndarray) -> np.ndarray:
    if len(grid) < 2:
        return np.empty((0, 4, 3))
    return np.stack(
        [grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]],
        axis=2,
    ).reshape((-1, 4, 3))


def _ring_element_velocity(
    points: np.ndarray,
    rings: np.ndarray,
    strengths: np.ndarray,
    core_radius: float,
) -> np.ndarray:
    if len(rings) == 0:
        return np.zeros_like(points)
    starts = np.concatenate([rings[:, 0], rings[:, 1], rings[:, 2], rings[:, 3]], axis=0)
    ends = np.concatenate([rings[:, 1], rings[:, 2], rings[:, 3], rings[:, 0]], axis=0)
    return _finite_segment_velocity_core(
        points,
        starts,
        ends,
        np.tile(np.asarray(strengths).reshape(-1), 4),
        core_radius,
    )


def _ring_influence_matrix(
    points: np.ndarray,
    normals: np.ndarray,
    rings: np.ndarray,
    core_radius: float,
) -> np.ndarray:
    columns = []
    for ring in rings:
        velocity = _ring_element_velocity(points, ring[None, :, :], np.ones(1), core_radius)
        columns.append(np.einsum("ij,ij->i", velocity, normals))
    return np.column_stack(columns)


def _dipole_element_velocity(
    points: np.ndarray,
    centers: np.ndarray,
    moments: np.ndarray,
    core_radius: float,
    exclude_self: bool = False,
) -> np.ndarray:
    if len(centers) == 0:
        return np.zeros_like(points)
    r = points[:, None, :] - centers[None, :, :]
    radius_sq = np.einsum("ijk,ijk->ij", r, r) + core_radius**2
    inverse_r3 = radius_sq ** -1.5
    inverse_r5 = radius_sq ** -2.5
    projection = np.einsum("ijk,jk->ij", r, moments)
    field = 3.0 * r * projection[:, :, None] * inverse_r5[:, :, None]
    field -= moments[None, :, :] * inverse_r3[:, :, None]
    if exclude_self:
        if len(points) != len(centers):
            raise ValueError("self exclusion requires matching point and dipole counts")
        field[np.arange(len(points)), np.arange(len(centers))] = 0.0
    return np.sum(field, axis=1) / (4.0 * math.pi)


def _trailing_edge_nodes(mesh: AeroMesh, n_span: int, n_chord: int) -> np.ndarray:
    return mesh.vertices.reshape((n_span + 1, n_chord + 1, 3))[:, -1].copy()


def _trailing_strengths(gamma: np.ndarray, n_span: int, n_chord: int) -> np.ndarray:
    # The new wake ring shares the trailing-edge segment with the bound ring
    # in the opposite geometrical direction, so equal circulation cancels the
    # duplicate segment and preserves a continuous vortex system.
    return gamma.reshape((n_span, n_chord))[:, -1].copy()


def _solve_unsteady_bound_circulation(
    influence: np.ndarray,
    mesh: AeroMesh,
    freestream: np.ndarray,
    wake_velocity: np.ndarray,
    regularization: float = 1e-8,
) -> np.ndarray:
    rhs = -np.einsum("ij,ij->i", mesh.normals, freestream[None, :] + wake_velocity)
    return np.linalg.lstsq(influence + np.eye(len(influence)) * regularization, rhs, rcond=None)[0]


def _normalised_moment_update(
    centers: np.ndarray,
    moments: np.ndarray,
    velocity_at,
    timestep: float,
    difference_step: float,
) -> np.ndarray:
    """Rotate dipole moments with the local velocity gradient at fixed magnitude."""
    if len(centers) == 0:
        return moments
    jacobian = np.zeros((len(centers), 3, 3), dtype=float)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = difference_step
        plus = velocity_at(centers + delta)
        minus = velocity_at(centers - delta)
        jacobian[:, :, axis] = (plus - minus) / (2.0 * difference_step)
    derivative = np.einsum("ijk,ik->ij", jacobian, moments)
    magnitude = np.linalg.norm(moments, axis=1)
    unit = moments / np.maximum(magnitude[:, None], 1e-12)
    derivative -= unit * np.einsum("ij,ij->i", derivative, unit)[:, None]
    updated = moments + timestep * derivative
    updated_norm = np.linalg.norm(updated, axis=1)
    return updated * (magnitude / np.maximum(updated_norm, 1e-12))[:, None]


def simulate_prescribed_free_wake(
    case: SimulationCase,
    method: str,
    core_radius_fraction: float = 0.08,
) -> WakeHistory:
    """Run an explicit vortex-ring or point-dipole wake reconstruction.

    The wing is represented by closed quadrilateral vortex rings so the shed
    wake is not double-counted. A dipole carries the circulation-weighted area
    vector of the ring it replaces. ``dipole_gradient`` rotates this moment with
    the local velocity gradient while preserving its magnitude, matching the
    vortex-particle evolution used by the point-dipole treatments.
    """
    if method not in {"panel", "dipole_fixed", "dipole_gradient"}:
        raise ValueError("method must be panel, dipole_fixed or dipole_gradient")
    if case.n_steps < 2:
        raise ValueError("case.n_steps must be at least 2")

    start = time.perf_counter()
    mesh = generate_aero_mesh(case.geometry)
    n_span = case.geometry.n_span
    n_chord = case.geometry.n_chord
    core_radius = core_radius_fraction * case.geometry.root_chord
    freestream_axis, _, _ = aerodynamic_axes(case.alpha_deg, mesh.body_axes)
    freestream = case.velocity * freestream_axis
    wing_rings = mesh.vertices[mesh.panels]
    wing_influence = _ring_influence_matrix(
        mesh.collocation_points,
        mesh.normals,
        wing_rings,
        core_radius,
    )
    gamma = _solve_unsteady_bound_circulation(
        wing_influence,
        mesh,
        freestream,
        np.zeros_like(mesh.collocation_points),
    )
    trailing_edge = _trailing_edge_nodes(mesh, n_span, n_chord)

    wake_grid = trailing_edge[None, :, :]
    wake_strengths = np.empty((0, n_span), dtype=float)
    dipole_centers = np.empty((0, 3), dtype=float)
    dipole_moments = np.empty((0, 3), dtype=float)
    dipole_strengths = np.empty(0, dtype=float)
    cl_history: list[float] = []
    downwash_history: list[float] = []
    centroid_history: list[np.ndarray] = []
    moment_history: list[float] = []
    area_history: list[float] = []
    finite_history: list[bool] = []

    for _ in range(case.n_steps):
        trailing_gamma = _trailing_strengths(gamma, n_span, n_chord)
        # Both representations begin from exactly the same geometrical ring at
        # the same quarter-step time level before their older wakes are moved.
        newborn_offset = 0.25 * case.timestep * freestream
        newborn_row = trailing_edge + newborn_offset
        newborn_rings = np.stack(
            [
                trailing_edge[:-1],
                trailing_edge[1:],
                trailing_edge[1:] + newborn_offset,
                trailing_edge[:-1] + newborn_offset,
            ],
            axis=1,
        )
        newborn_centers, newborn_moments = ring_to_equivalent_dipole(
            newborn_rings,
            trailing_gamma,
        )
        if method == "panel":
            existing_rows = wake_grid[1:]
            if len(existing_rows):
                points = existing_rows.reshape((-1, 3))
                velocity = np.broadcast_to(freestream, points.shape).copy()
                velocity += _ring_element_velocity(points, wing_rings, gamma, core_radius)
                existing_rings = _grid_rings(wake_grid)
                velocity += _ring_element_velocity(
                    points,
                    existing_rings,
                    wake_strengths.reshape(-1),
                    core_radius,
                )
                existing_rows = existing_rows + case.timestep * velocity.reshape(existing_rows.shape)
            wake_grid = np.concatenate(
                [trailing_edge[None, :, :], newborn_row[None, :, :], existing_rows],
                axis=0,
            )
            wake_strengths = np.concatenate([trailing_gamma[None, :], wake_strengths], axis=0)
            wake_rings = _grid_rings(wake_grid)
            wake_velocity = _ring_element_velocity(
                mesh.collocation_points,
                wake_rings,
                wake_strengths.reshape(-1),
                core_radius,
            )
            centers = wake_rings.mean(axis=1)
            moments = wake_strengths.reshape(-1, 1) * _ring_area_vectors(wake_rings)
            ring_areas = np.linalg.norm(_ring_area_vectors(wake_rings), axis=1)
        else:
            def velocity_at(points: np.ndarray) -> np.ndarray:
                value = np.broadcast_to(freestream, points.shape).copy()
                value += _ring_element_velocity(points, wing_rings, gamma, core_radius)
                value += _dipole_element_velocity(points, dipole_centers, dipole_moments, core_radius)
                return value

            if len(dipole_centers):
                velocity = np.broadcast_to(freestream, dipole_centers.shape).copy()
                velocity += _ring_element_velocity(dipole_centers, wing_rings, gamma, core_radius)
                velocity += _dipole_element_velocity(
                    dipole_centers,
                    dipole_centers,
                    dipole_moments,
                    core_radius,
                    exclude_self=True,
                )
                if method == "dipole_gradient":
                    dipole_moments = _normalised_moment_update(
                        dipole_centers,
                        dipole_moments,
                        velocity_at,
                        case.timestep,
                        0.025 * case.geometry.root_chord,
                    )
                dipole_centers += case.timestep * velocity
            dipole_centers = np.concatenate([newborn_centers, dipole_centers], axis=0)
            dipole_moments = np.concatenate([newborn_moments, dipole_moments], axis=0)
            dipole_strengths = np.concatenate([trailing_gamma, dipole_strengths], axis=0)
            wake_velocity = _dipole_element_velocity(
                mesh.collocation_points,
                dipole_centers,
                dipole_moments,
                core_radius,
            )
            centers = dipole_centers
            moments = dipole_moments
            ring_areas = np.linalg.norm(moments, axis=1) / np.maximum(np.abs(dipole_strengths), 1e-12)

        gamma = _solve_unsteady_bound_circulation(
            wing_influence,
            mesh,
            freestream,
            wake_velocity,
        )
        cl, _ = circulation_lift_coefficient(mesh, case, gamma, freestream)
        cl_history.append(cl)
        downwash_history.append(float(np.sqrt(np.mean(wake_velocity[:, 2] ** 2))))
        centroid_history.append(np.mean(centers, axis=0))
        # Track one fixed cohort: the oldest row, shed at the first step.
        moment_history.append(float(np.mean(np.linalg.norm(moments[-n_span:], axis=1))))
        area_history.append(float(np.min(ring_areas)))
        finite_history.append(
            bool(
                np.all(np.isfinite(gamma))
                and np.all(np.isfinite(centers))
                and np.all(np.isfinite(moments))
                and np.all(np.isfinite(wake_velocity))
            )
        )

    return WakeHistory(
        method=method,
        element_centers=centers,
        moments=moments,
        cl=np.asarray(cl_history),
        downwash_rms=np.asarray(downwash_history),
        centroid=np.asarray(centroid_history),
        moment_mean=np.asarray(moment_history),
        minimum_area=np.asarray(area_history),
        finite_state=np.asarray(finite_history, dtype=bool),
        runtime_s=time.perf_counter() - start,
        grid=wake_grid if method == "panel" else None,
    )


def compare_wake_histories(panel: WakeHistory, dipole: WakeHistory, root_chord: float) -> dict[str, float]:
    if panel.element_centers.shape != dipole.element_centers.shape:
        raise ValueError("wake histories must contain matching shed elements")
    displacement = np.linalg.norm(panel.element_centers - dipole.element_centers, axis=1) / root_chord
    centroid_delta = np.linalg.norm(panel.centroid - dipole.centroid, axis=1) / root_chord
    cl_delta = np.abs(panel.cl - dipole.cl)
    downwash_delta = np.abs(panel.downwash_rms - dipole.downwash_rms)
    panel_moment_drift = abs(panel.moment_mean[-1] - panel.moment_mean[0]) / max(panel.moment_mean[0], 1e-12)
    dipole_moment_drift = abs(dipole.moment_mean[-1] - dipole.moment_mean[0]) / max(dipole.moment_mean[0], 1e-12)
    return {
        "trajectory_rms_chords": float(np.sqrt(np.mean(displacement**2))),
        "trajectory_max_chords": float(np.max(displacement)),
        "centroid_final_delta_chords": float(centroid_delta[-1]),
        "cl_mean_abs_delta": float(np.mean(cl_delta)),
        "cl_final_abs_delta": float(cl_delta[-1]),
        "downwash_mean_abs_delta": float(np.mean(downwash_delta)),
        "panel_moment_relative_drift": float(panel_moment_drift),
        "dipole_moment_relative_drift": float(dipole_moment_drift),
        "panel_minimum_ring_area": float(np.min(panel.minimum_area)),
        "dipole_minimum_equivalent_area": float(np.min(dipole.minimum_area)),
        "panel_finite_fraction": float(np.mean(panel.finite_state)),
        "dipole_finite_fraction": float(np.mean(dipole.finite_state)),
        "panel_runtime_s": float(panel.runtime_s),
        "dipole_runtime_s": float(dipole.runtime_s),
    }
