from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np

from run_pipeline import (
    AVL,
    OUT,
    _write_csv,
    run_solver_validation,
    solver_validation_gates,
)


def main() -> None:
    if not AVL.exists():
        raise FileNotFoundError("Run scripts/fetch_avl_reference.sh before the independent benchmark")

    start = time.perf_counter()
    rows = run_solver_validation()
    gates = solver_validation_gates(rows)
    runtime = time.perf_counter() - start
    _write_csv(OUT / "solver_validation.csv", rows)

    metadata = {
        "experiment": "independent steady aerodynamic validation",
        "cases": len(rows),
        "aspect_ratios": [4, 8, 12, 20],
        "angles_of_attack_deg": [2.0, 5.0, 8.0],
        "lattice": {"spanwise_panels": 20, "chordwise_panels": 4},
        "lift_and_moment_method": "Kutta-Joukowski force on bound vortex segments",
        "induced_drag_method": "discrete Trefftz-plane trailing-vortex sheet",
        "reference": "MIT AVL 3.32 independent implementation reference",
        "avl_executable": str(AVL),
        "avl_version": rows[0]["avl_version"],
        "runtime_s": runtime,
        "validation_gates": gates,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    path = OUT / "solver_validation_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "solver_validation_gates.json").write_text(
        json.dumps(gates, indent=2), encoding="utf-8"
    )
    if not gates["passed"]["cl"]:
        raise RuntimeError("rectangular-wing AVL CL MAE exceeds 0.02")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
