from __future__ import annotations
import math
import time

import numpy as np

from .geometry import generate_aero_mesh
from .types import AeroMesh, AeroResult, SimulationCase


BODY_X = np.array([1.0, 0.0, 0.0])  # chordwise, leading edge to trailing edge
BODY_Y = np.array([0.0, 1.0, 0.0])  # spanwise, left tip to right tip
BODY_Z = np.array([0.0, 0.0, 1.0])  # upward


def aerodynamic_axes(
    alpha_deg: float,
    body_axes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit freestream, wind-axis lift and spanwise moment vectors.

    Positive angle of attack gives the incoming flow a positive body-z
    component. Positive circulation runs from lower to higher body-y and
    therefore produces positive lift. Positive pitching moment is nose-up.
    """
    alpha = math.radians(alpha_deg)
    transform = np.eye(3) if body_axes is None else np.asarray(body_axes, dtype=float)
    if transform.shape != (3, 3):
        raise ValueError("body_axes must be a 3 by 3 matrix")
    freestream = transform @ np.array([math.cos(alpha), 0.0, math.sin(alpha)])
    lift = transform @ np.array([-math.sin(alpha), 0.0, math.cos(alpha)])
    pitch = transform @ BODY_Y
    return (
        freestream / max(float(np.linalg.norm(freestream)), 1e-12),
        lift / max(float(np.linalg.norm(lift)), 1e-12),
        pitch / max(float(np.linalg.norm(pitch)), 1e-12),
    )


def finite_segment_velocity(point: np.ndarray, a: np.ndarray, b: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    r1 = point - a
    r2 = point - b
    r0 = b - a
    cross = np.cross(r1, r2)
    cross_norm_sq = float(np.dot(cross, cross))
    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))
    if cross_norm_sq < 1e-14 or r1_norm < 1e-12 or r2_norm < 1e-12:
        return np.zeros(3)
    factor = gamma / (4.0 * math.pi * cross_norm_sq)
    factor *= float(np.dot(r0, (r1 / r1_norm) - (r2 / r2_norm)))
    return factor * cross


def horseshoe_velocity(
    point: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    wake_length: float,
    gamma: float = 1.0,
    downstream_direction: np.ndarray | None = None,
) -> np.ndarray:
    direction = np.array([1.0, 0.0, 0.0]) if downstream_direction is None else np.asarray(downstream_direction)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    downstream = wake_length * direction
    left_far = left + downstream
    right_far = right + downstream
    return (
        finite_segment_velocity(point, left, right, gamma)
        + finite_segment_velocity(point, right, right_far, gamma)
        + finite_segment_velocity(point, left_far, left, gamma)
    )


def finite_segment_velocity_matrix(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized unit-circulation Biot-Savart influence for point/segment pairs."""
    r1 = points[:, None, :] - a[None, :, :]
    r2 = points[:, None, :] - b[None, :, :]
    r0 = b - a
    cross = np.cross(r1, r2)
    cross_norm_sq = np.einsum("ijk,ijk->ij", cross, cross)
    r1_norm = np.linalg.norm(r1, axis=2)
    r2_norm = np.linalg.norm(r2, axis=2)
    valid = (cross_norm_sq >= 1e-14) & (r1_norm >= 1e-12) & (r2_norm >= 1e-12)
    direction_difference = r1 / np.maximum(r1_norm[:, :, None], 1e-12)
    direction_difference -= r2 / np.maximum(r2_norm[:, :, None], 1e-12)
    projection = np.einsum("jk,ijk->ij", r0, direction_difference)
    factor = projection / np.maximum(4.0 * math.pi * cross_norm_sq, 1e-14)
    factor[~valid] = 0.0
    return factor[:, :, None] * cross


def horseshoe_influence_matrix(
    points: np.ndarray,
    normals: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    wake_length: float,
    downstream_direction: np.ndarray | None = None,
) -> np.ndarray:
    direction = np.array([1.0, 0.0, 0.0]) if downstream_direction is None else np.asarray(downstream_direction)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    downstream = wake_length * direction
    left_far = left + downstream
    right_far = right + downstream
    velocities = finite_segment_velocity_matrix(points, left, right)
    velocities += finite_segment_velocity_matrix(points, right, right_far)
    velocities += finite_segment_velocity_matrix(points, left_far, left)
    return np.einsum("ijk,ik->ij", velocities, normals)


def solve_steady_vlm(
    case: SimulationCase,
    regularization: float = 1e-8,
    wake_length_factor: float = 1.0,
) -> AeroResult:
    start = time.perf_counter()
    mesh = generate_aero_mesh(case.geometry)
    result = solve_mesh_steady_vlm(mesh, case, regularization, wake_length_factor)
    result.runtime_s = time.perf_counter() - start
    return result


def solve_bound_circulation(
    mesh: AeroMesh,
    case: SimulationCase,
    external_velocity: np.ndarray | None = None,
    regularization: float = 1e-8,
    wake_length_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Solve bound circulation with an optional velocity field at collocation points."""
    freestream_axis, _, _ = aerodynamic_axes(case.alpha_deg, mesh.body_axes)
    v_inf = case.velocity * freestream_axis
    if wake_length_factor <= 0.0:
        raise ValueError("wake_length_factor must be positive")
    wake_length = wake_length_factor * max(50.0 * mesh.reference_chord, 10.0 * mesh.reference_span)
    influence = horseshoe_influence_matrix(
        mesh.collocation_points,
        mesh.normals,
        mesh.bound_left,
        mesh.bound_right,
        wake_length,
        downstream_direction=v_inf,
    )
    total_external = np.broadcast_to(v_inf, mesh.collocation_points.shape).copy()
    if external_velocity is not None:
        if external_velocity.shape != mesh.collocation_points.shape:
            raise ValueError("external_velocity must match mesh.collocation_points")
        total_external += external_velocity
    rhs = -np.einsum("ij,ij->i", mesh.normals, total_external)
    influence += np.eye(len(mesh.panels)) * regularization
    converged = True
    try:
        gamma = np.linalg.solve(influence, rhs)
    except np.linalg.LinAlgError:
        gamma = np.linalg.lstsq(influence, rhs, rcond=None)[0]
        converged = False
    return gamma, v_inf, converged


def circulation_lift_coefficient(
    mesh: AeroMesh,
    case: SimulationCase,
    gamma: np.ndarray,
    freestream: np.ndarray,
) -> tuple[float, float]:
    """Return lift and pitching-moment coefficients from bound circulation."""
    forces = case.density * gamma[:, None] * np.cross(
        np.broadcast_to(freestream, mesh.bound_right.shape),
        mesh.bound_right - mesh.bound_left,
    )
    _, lift_axis, pitch_axis = aerodynamic_axes(case.alpha_deg, mesh.body_axes)
    total_force = np.sum(forces, axis=0)
    lift = float(np.dot(total_force, lift_axis))
    centers = 0.5 * (mesh.bound_left + mesh.bound_right)
    total_moment = np.sum(np.cross(centers - mesh.reference_point, forces), axis=0)
    pitching_moment = float(np.dot(total_moment, pitch_axis))
    q = 0.5 * case.density * case.velocity**2
    cl = lift / max(q * mesh.reference_area, 1e-12)
    cm = pitching_moment / max(q * mesh.reference_area * mesh.reference_chord, 1e-12)
    return float(cl), float(cm)


def trefftz_induced_drag_coefficient(
    mesh: AeroMesh,
    case: SimulationCase,
    gamma: np.ndarray,
) -> float:
    """Calculate steady induced drag from the discrete trailing-vortex sheet."""
    n_span = case.geometry.n_span
    n_chord = case.geometry.n_chord
    if gamma.shape != (n_span * n_chord,):
        raise ValueError("gamma must contain one circulation value per aerodynamic panel")

    circulation = gamma.reshape(n_span, n_chord).sum(axis=1)
    bound_left = mesh.bound_left.reshape(n_span, n_chord, 3)
    bound_right = mesh.bound_right.reshape(n_span, n_chord, 3)
    span_axis = mesh.body_axes[:, 1]
    boundaries = np.concatenate(
        [bound_left[:, 0] @ span_axis, np.asarray([bound_right[-1, 0] @ span_axis])]
    )
    centres = 0.5 * (boundaries[:-1] + boundaries[1:])
    widths = np.diff(boundaries)

    trailing_strength = np.empty(n_span + 1, dtype=float)
    trailing_strength[0] = circulation[0]
    trailing_strength[1:-1] = np.diff(circulation)
    trailing_strength[-1] = -circulation[-1]

    distances = centres[:, None] - boundaries[None, :]
    downwash = np.sum(trailing_strength[None, :] / distances, axis=1) / (4.0 * math.pi)
    induced_drag = case.density * float(np.sum(circulation * downwash * widths))
    q = 0.5 * case.density * case.velocity**2
    return abs(induced_drag) / max(q * mesh.reference_area, 1e-12)


def solve_mesh_steady_vlm(
    mesh: AeroMesh,
    case: SimulationCase,
    regularization: float = 1e-8,
    wake_length_factor: float = 1.0,
) -> AeroResult:
    start = time.perf_counter()
    gamma, v_inf, converged = solve_bound_circulation(
        mesh,
        case,
        regularization=regularization,
        wake_length_factor=wake_length_factor,
    )
    cl, cm = circulation_lift_coefficient(mesh, case, gamma, v_inf)
    cdi = trefftz_induced_drag_coefficient(mesh, case, gamma)

    return AeroResult(
        case_name=case.name,
        cl=float(cl),
        cdi=float(cdi),
        cm=float(cm),
        gamma_norm=float(np.linalg.norm(gamma)),
        runtime_s=time.perf_counter() - start,
        converged=converged,
    )


def prandtl_lift_coefficient(alpha_deg: float, aspect_ratio: float, oswald_efficiency: float = 1.0) -> float:
    alpha = math.radians(alpha_deg)
    a0 = 2.0 * math.pi
    slope = a0 / (1.0 + a0 / (math.pi * oswald_efficiency * aspect_ratio))
    return slope * alpha
