from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import aerodynamic_reference, chord_at_y, leading_edge_x, naca_00xx_thickness
from .types import AeroMesh, WingParameters


@dataclass(frozen=True)
class AvlResult:
    cl: float
    cdi: float
    cm: float
    version: str


def _airfoil_coordinates(params: WingParameters, twist_deg: float, points: int = 41) -> list[str]:
    """Return the exact section used by the internal geometry as AVL AIRFOIL data."""
    x_forward = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, points)))
    mean = 4.0 * params.camber_ratio * x_forward * (1.0 - x_forward)
    mean -= np.tan(np.deg2rad(twist_deg)) * (x_forward - 0.25)
    thickness = naca_00xx_thickness(x_forward, params.thickness_ratio)
    x = np.concatenate([x_forward[::-1], x_forward[1:]])
    z = np.concatenate([(mean + thickness)[::-1], (mean - thickness)[1:]])
    return [f"{xi:.10f} {zi:.10f}" for xi, zi in zip(x, z)]


def _section_locations(params: WingParameters) -> np.ndarray:
    half_span = params.span / 2.0
    if params.planform_family == "elliptical":
        return np.linspace(0.0, half_span, 9)
    if params.planform_family == "cranked":
        return np.array([0.0, params.kink_span_fraction * half_span, half_span])
    return np.array([0.0, half_span])


def _airfoil_file_text(params: WingParameters, twist_deg: float) -> str:
    return "\n".join([params.name, *_airfoil_coordinates(params, twist_deg), ""])


def avl_geometry_text(
    params: WingParameters,
    airfoil_filenames: list[str] | None = None,
) -> str:
    sections = []
    half_span = params.span / 2.0
    section_locations = _section_locations(params)
    if airfoil_filenames is None:
        airfoil_filenames = [f"section-{index:02d}.dat" for index in range(len(section_locations))]
    if len(airfoil_filenames) != len(section_locations):
        raise ValueError("one airfoil file is required for each AVL section")
    for index, y in enumerate(section_locations):
        y_array = np.array([y])
        chord = float(chord_at_y(params, y_array)[0])
        leading = float(leading_edge_x(params, y_array)[0])
        z = float(np.tan(np.deg2rad(params.dihedral_deg)) * y)
        twist = float(params.twist_deg * (y / max(half_span, 1e-12)))
        sections.append(
            "\n".join(
                [
                    "SECTION",
                    f"{leading:.10f} {y:.10f} {z:.10f} {chord:.10f} 0.0000000000",
                    "AFILE",
                    f"{airfoil_filenames[index]} 0.0 1.0",
                ]
            )
        )
    reference_area, reference_chord, reference_span, reference_point = aerodynamic_reference(params)
    return "\n".join(
        [
            params.name,
            "0.0",
            "0 0 0.0",
            f"{reference_area:.10f} {reference_chord:.10f} {reference_span:.10f}",
            f"{reference_point[0]:.10f} {reference_point[1]:.10f} {reference_point[2]:.10f}",
            "0.0",
            "SURFACE",
            "Wing",
            f"{params.n_chord} 1.0 {max(params.n_span // 2, 2 * (len(section_locations) - 1), 4)} -1.0",
            "YDUPLICATE",
            "0.0",
            *sections,
            "SCALE",
            "1.0 1.0 1.0",
            "",
        ]
    )


def _extract_value(text: str, name: str) -> float:
    match = re.search(rf"\b{name}\s*=\s*([-+0-9.Ee]+)", text)
    if not match:
        raise ValueError(f"AVL output did not contain {name}")
    return float(match.group(1))


def run_avl_reference(
    params: WingParameters,
    alpha_deg: float,
    executable: str | Path,
    timeout_s: float = 20.0,
) -> AvlResult:
    executable_path = Path(executable).resolve()
    if not executable_path.exists():
        raise FileNotFoundError(executable_path)
    with tempfile.TemporaryDirectory(prefix="aero-avl-") as directory:
        root = Path(directory)
        geometry_path = root / "case.avl"
        output_path = root / "forces.txt"
        section_locations = _section_locations(params)
        airfoil_filenames = [f"section-{index:02d}.dat" for index in range(len(section_locations))]
        geometry_path.write_text(avl_geometry_text(params, airfoil_filenames), encoding="ascii")
        half_span = params.span / 2.0
        for filename, y in zip(airfoil_filenames, section_locations):
            twist = float(params.twist_deg * (y / max(half_span, 1e-12)))
            (root / filename).write_text(_airfoil_file_text(params, twist), encoding="ascii")
        commands = "\n".join(
            [
                "OPER",
                "A",
                "A",
                f"{alpha_deg:.10f}",
                "X",
                "FT",
                output_path.name,
                "",
                "QUIT",
                "",
            ]
        )
        process = subprocess.run(
            [str(executable_path), geometry_path.name],
            input=commands,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            cwd=root,
        )
        if process.returncode != 0 or not output_path.exists():
            diagnostic = (process.stdout + "\n" + process.stderr)[-3000:]
            raise RuntimeError(f"AVL failed ({process.returncode}): {diagnostic}")
        output = output_path.read_text(encoding="ascii", errors="replace")
        banner = process.stdout + process.stderr
        version_match = re.search(r"Version\s+([0-9.]+)", banner)
        version = version_match.group(1) if version_match else "unknown"
        if version != "3.32":
            raise RuntimeError(f"Expected AVL 3.32, found {version}")
        return AvlResult(
            cl=_extract_value(output, "CLtot"),
            cdi=_extract_value(output, "CDind"),
            cm=_extract_value(output, "Cmtot"),
            version=version,
        )


def _mesh_section_airfoil(grid_section: np.ndarray) -> str:
    leading = grid_section[0]
    chord = max(float(grid_section[-1, 0] - leading[0]), 1e-12)
    x = np.clip((grid_section[:, 0] - leading[0]) / chord, 0.0, 1.0)
    mean = (grid_section[:, 2] - leading[2]) / chord
    epsilon = 1e-7
    loop_x = np.concatenate([x[::-1], x[1:]])
    loop_z = np.concatenate([(mean + epsilon)[::-1], (mean - epsilon)[1:]])
    return "\n".join(["Reconstructed section", *[f"{xi:.10f} {zi:.10f}" for xi, zi in zip(loop_x, loop_z)], ""])


def avl_mesh_geometry_text(mesh: AeroMesh, airfoil_filenames: list[str]) -> str:
    """Represent a reconstructed structured lattice with matching AVL sections."""
    n_span = int(mesh.metadata.get("n_span", 0))
    n_chord = int(mesh.metadata.get("n_chord", 0))
    expected_vertices = (n_span + 1) * (n_chord + 1)
    if n_span < 1 or n_chord < 1 or len(mesh.vertices) != expected_vertices:
        raise ValueError("mesh must be a structured span-by-chord lattice")
    if len(airfoil_filenames) != n_span + 1:
        raise ValueError("one airfoil file is required for each span station")
    grid = mesh.vertices.reshape((n_span + 1, n_chord + 1, 3))
    sections = []
    for station, filename in zip(grid, airfoil_filenames):
        leading = station[0]
        chord = float(station[-1, 0] - station[0, 0])
        if chord <= 0.0:
            raise ValueError("AVL section has non-positive projected chord")
        sections.append(
            "\n".join(
                [
                    "SECTION",
                    f"{leading[0]:.10f} {leading[1]:.10f} {leading[2]:.10f} {chord:.10f} 0.0000000000",
                    "AFILE",
                    f"{filename} 0.0 1.0",
                ]
            )
        )
    return "\n".join(
        [
            "Reconstructed lifting surface",
            "0.0",
            "0 0 0.0",
            f"{mesh.reference_area:.10f} {mesh.reference_chord:.10f} {mesh.reference_span:.10f}",
            f"{mesh.reference_point[0]:.10f} {mesh.reference_point[1]:.10f} {mesh.reference_point[2]:.10f}",
            "0.0",
            "SURFACE",
            "Wing",
            f"{n_chord} 1.0 {max(2 * n_span, 20)} -1.0",
            *sections,
            "SCALE",
            "1.0 1.0 1.0",
            "",
        ]
    )


def run_avl_mesh_reference(
    mesh: AeroMesh,
    alpha_deg: float,
    executable: str | Path,
    timeout_s: float = 30.0,
) -> AvlResult:
    """Run AVL on the same structured geometry and reference quantities as the internal solver."""
    executable_path = Path(executable).resolve()
    if not executable_path.exists():
        raise FileNotFoundError(executable_path)
    n_span = int(mesh.metadata.get("n_span", 0))
    n_chord = int(mesh.metadata.get("n_chord", 0))
    grid = mesh.vertices.reshape((n_span + 1, n_chord + 1, 3))
    with tempfile.TemporaryDirectory(prefix="aero-avl-mesh-") as directory:
        root = Path(directory)
        geometry_path = root / "case.avl"
        output_path = root / "forces.txt"
        filenames = [f"section-{index:03d}.dat" for index in range(n_span + 1)]
        geometry_path.write_text(avl_mesh_geometry_text(mesh, filenames), encoding="ascii")
        for filename, station in zip(filenames, grid):
            (root / filename).write_text(_mesh_section_airfoil(station), encoding="ascii")
        commands = "\n".join(
            ["OPER", "A", "A", f"{alpha_deg:.10f}", "X", "FT", output_path.name, "", "QUIT", ""]
        )
        process = subprocess.run(
            [str(executable_path), geometry_path.name],
            input=commands,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            cwd=root,
        )
        if process.returncode != 0 or not output_path.exists():
            diagnostic = (process.stdout + "\n" + process.stderr)[-3000:]
            raise RuntimeError(f"AVL failed ({process.returncode}): {diagnostic}")
        output = output_path.read_text(encoding="ascii", errors="replace")
        banner = process.stdout + process.stderr
        version_match = re.search(r"Version\s+([0-9.]+)", banner)
        version = version_match.group(1) if version_match else "unknown"
        if version != "3.32":
            raise RuntimeError(f"Expected AVL 3.32, found {version}")
        return AvlResult(
            cl=_extract_value(output, "CLtot"),
            cdi=_extract_value(output, "CDind"),
            cm=_extract_value(output, "Cmtot"),
            version=version,
        )
