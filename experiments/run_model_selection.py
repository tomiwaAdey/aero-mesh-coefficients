"""Export the Random Forest part of the principal validation-only selection.

Hyperparameters are selected inside the principal experiment so every model uses
the same geometry splits and end-to-end ranking. This compatibility entry point
only produces the model-specific audit tables; it does not run a second search.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def main() -> None:
    metadata_path = OUT / "experiment_metadata.json"
    selection_path = OUT / "model_selection.csv"
    cases_path = OUT / "model_selection_by_geometry.csv"
    for path in (metadata_path, selection_path, cases_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; run experiments/run_pipeline.py first"
            )

    experiment = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(selection_path)
    cases = pd.read_csv(cases_path)
    summary = summary[summary["model"] == "random_forest"].copy()
    cases = cases[cases["model"] == "random_forest"].copy()
    if summary.empty or cases.empty:
        raise RuntimeError("The principal model selection contains no Random Forest candidates")

    summary.to_csv(OUT / "rf_hyperparameter_selection_summary.csv", index=False)
    cases.to_csv(OUT / "rf_hyperparameter_selection_by_geometry.csv", index=False)
    metadata = {
        "source": "principal validation-only model selection",
        "selection_order": experiment["selection_order"],
        "dataset_seeds": experiment["dataset_seeds"],
        "selected_random_forest_parameters": {
            str(seed): parameters["random_forest"]
            for seed, parameters in experiment["selected_hyperparameters"].items()
        },
        "failure_penalty": None,
    }
    (OUT / "rf_hyperparameter_selection_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
