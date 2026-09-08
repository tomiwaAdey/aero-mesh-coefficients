from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}

MODEL_PARAMETER_GRIDS: dict[str, tuple[dict[str, object], ...]] = {
    "logistic": ({"C": 0.5}, {"C": 2.0}),
    "knn": ({"n_neighbors": 7}, {"n_neighbors": 13}),
    "svm_rbf": ({"C": 6.0, "gamma": "scale"}, {"C": 12.0, "gamma": "scale"}),
    "random_forest": (
        {"max_features": "sqrt", "min_samples_leaf": 2},
        {"max_features": 0.5, "min_samples_leaf": 2},
    ),
    "extra_trees": (
        {"max_features": "sqrt", "min_samples_leaf": 2},
        {"max_features": 0.5, "min_samples_leaf": 2},
    ),
}


@dataclass
class ClassificationResult:
    model_name: str
    metrics: dict[str, float]
    confusion: np.ndarray
    predictions: np.ndarray


def build_model(name: str, seed: int = 13, parameters: dict[str, object] | None = None) -> object:
    selected = dict(parameters or {})

    def logistic():
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=float(selected.get("C", 2.0)),
                        random_state=seed,
                    ),
                ),
            ]
        )

    def knn():
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=int(selected.get("n_neighbors", 13)), weights="distance"
                    ),
                ),
            ]
        )

    def random_forest():
        return RandomForestClassifier(
            **{
                **RANDOM_FOREST_PARAMETERS,
                "max_features": selected.get("max_features", RANDOM_FOREST_PARAMETERS["max_features"]),
                "min_samples_leaf": int(
                    selected.get("min_samples_leaf", RANDOM_FOREST_PARAMETERS["min_samples_leaf"])
                ),
            },
            random_state=seed,
        )

    def extra_trees():
        return ExtraTreesClassifier(
            n_estimators=300,
            max_features=selected.get("max_features", "sqrt"),
            min_samples_leaf=int(selected.get("min_samples_leaf", 2)),
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )

    builders = {
        "logistic": logistic,
        "knn": knn,
        "svm_rbf": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=float(selected.get("C", 12.0)),
                        gamma=selected.get("gamma", "scale"),
                        class_weight="balanced",
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": random_forest,
        "extra_trees": extra_trees,
    }
    if name not in builders:
        raise KeyError(name)
    return builders[name]()


def model_catalog(seed: int = 13) -> dict[str, object]:
    return {name: build_model(name, seed, grid[-1]) for name, grid in MODEL_PARAMETER_GRIDS.items()}


def predict_class_probabilities(model: object, features: np.ndarray) -> np.ndarray:
    """Return probabilities in the fixed surface/LE/TE/tip class order."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("model does not expose class probabilities")
    raw = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    result = np.full((len(features), 4), 1e-12, dtype=float)
    result[:, classes] = raw
    return result / result.sum(axis=1, keepdims=True)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    labels = np.array([0, 1, 2, 3], dtype=int)
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    for label, name in [(0, "surface"), (1, "leading"), (2, "trailing"), (3, "tip")]:
        metrics[f"f1_{name}"] = float(f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0))
    return metrics, confusion


def fit_and_evaluate_models(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int = 13,
) -> tuple[dict[str, object], list[ClassificationResult]]:
    fitted = {}
    results = []
    for name, model in model_catalog(seed).items():
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        metrics, confusion = classification_metrics(test_y, predictions)
        fitted[name] = model
        results.append(ClassificationResult(name, metrics, confusion, predictions))
    return fitted, results
