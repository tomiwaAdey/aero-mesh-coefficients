from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeometryInput:
    source: str
    path: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TriMesh:
    vertices: np.ndarray
    faces: np.ndarray
    labels: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalFrame:
    origin: np.ndarray
    axes: np.ndarray
    scale: float


@dataclass(frozen=True)
class WingParameters:
    name: str
    span: float
    root_chord: float
    tip_chord: float
    sweep_deg: float = 0.0
    dihedral_deg: float = 0.0
    twist_deg: float = 0.0
    planform_family: str = "trapezoidal"
    kink_span_fraction: float = 0.45
    kink_chord_ratio: float = 0.72
    thickness_ratio: float = 0.12
    camber_ratio: float = 0.0
    n_span: int = 16
    n_chord: int = 4

    @property
    def area(self) -> float:
        if self.planform_family == "elliptical":
            ellipse = np.pi * self.span * self.root_chord / 4.0
            rectangle = self.span * self.tip_chord
            return float(rectangle + (1.0 - self.taper_ratio) * ellipse)
        if self.planform_family == "cranked":
            half_span = self.span / 2.0
            kink_span = half_span * self.kink_span_fraction
            kink_chord = self.root_chord * self.kink_chord_ratio
            half_area = 0.5 * (self.root_chord + kink_chord) * kink_span
            half_area += 0.5 * (kink_chord + self.tip_chord) * (half_span - kink_span)
            return 2.0 * half_area
        return 0.5 * (self.root_chord + self.tip_chord) * self.span

    @property
    def aspect_ratio(self) -> float:
        return self.span**2 / self.area

    @property
    def taper_ratio(self) -> float:
        return self.tip_chord / self.root_chord


@dataclass
class GeometryFeatures:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray
    saliency: np.ndarray
    chord_fraction: np.ndarray
    span_fraction: np.ndarray
    labels: np.ndarray | None = None


@dataclass
class AeroMesh:
    vertices: np.ndarray
    panels: np.ndarray
    collocation_points: np.ndarray
    normals: np.ndarray
    bound_left: np.ndarray
    bound_right: np.ndarray
    span_widths: np.ndarray
    reference_area: float
    reference_chord: float
    reference_span: float
    reference_point: np.ndarray
    wake_edges: np.ndarray
    body_axes: np.ndarray = field(default_factory=lambda: np.eye(3))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationCase:
    name: str
    geometry: WingParameters
    alpha_deg: float
    velocity: float = 1.0
    density: float = 1.0
    timestep: float = 0.002
    n_steps: int = 1


@dataclass
class AeroResult:
    case_name: str
    cl: float
    cdi: float
    cm: float
    gamma_norm: float
    runtime_s: float
    converged: bool
    wake_metric: float | None = None


@dataclass
class ExperimentRun:
    name: str
    track: str
    seed: int
    metrics: dict[str, float]
    artifacts: list[Path]
