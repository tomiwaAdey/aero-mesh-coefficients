from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .stl import mirror_half_surface, repair_trimesh
from .types import TriMesh


def geometry_artifact_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _label_mapping(data) -> dict[tuple[int, int], int]:
    return {
        tuple(int(value) for value in pair): int(label)
        for pair, label in zip(data["source_edge_pairs"], data["source_edge_classes"])
    }


def load_onera_m6_artifact(path: str | Path) -> TriMesh:
    source = Path(path)
    data = np.load(source)
    actual_hash = geometry_artifact_hash(
        data["vertices"],
        data["faces"],
        data["source_edge_pairs"],
        data["source_edge_classes"],
    )
    expected_hash = str(data["geometry_sha256"])
    if actual_hash != expected_hash:
        raise ValueError(f"ONERA M6 geometry hash mismatch: {actual_hash} != {expected_hash}")
    metadata = {
        "source": str(source),
        "geometry": "ONERA M6",
        "geometry_sha256": actual_hash,
        "source_edge_labels": _label_mapping(data),
        "source_curves": {
            "leading_edge": data["source_leading_edge"].astype(float),
            "trailing_edge": data["source_trailing_edge"].astype(float),
            "tips": [data["source_tip_left"].astype(float), data["source_tip_right"].astype(float)],
        },
        "source_root_chord": float(data["source_root_chord"]),
        "reference_annotation_method": "preserved construction curves",
        "reference_annotations_are_algorithmic": False,
        "intentionally_open": False,
    }
    return repair_trimesh(
        TriMesh(data["vertices"].astype(float), data["faces"].astype(int), metadata=metadata),
        intentionally_open=False,
    )


def load_crm_lifting_surface_artifact(path: str | Path) -> TriMesh:
    source = Path(path)
    data = np.load(source)
    actual_hash = geometry_artifact_hash(
        data["vertices"],
        data["faces"],
        data["source_edge_pairs"],
        data["source_edge_classes"],
    )
    expected_hash = str(data["geometry_sha256"])
    if actual_hash != expected_hash:
        raise ValueError(f"CRM geometry hash mismatch: {actual_hash} != {expected_hash}")

    root_y = float(data["source_root_y"])
    vertices = data["vertices"].astype(float).copy()
    vertices[:, 1] -= root_y
    leading = data["source_leading_edge"].astype(float).copy()
    trailing = data["source_trailing_edge"].astype(float).copy()
    tip = data["source_tip"].astype(float).copy()
    for curve in (leading, trailing, tip):
        curve[:, 1] -= root_y
    metadata = {
        "source": str(source),
        "geometry": "NASA DPW4 high-speed CRM isolated lifting surface",
        "geometry_sha256": actual_hash,
        "source_edge_labels": _label_mapping(data),
        "source_curves": {
            "leading_edge": leading,
            "trailing_edge": trailing,
            "tips": [tip],
        },
        "source_root_chord": float(data["source_root_chord"]),
        "reference_annotation_method": str(data["reference_annotation_method"]),
        "reference_annotations_are_algorithmic": bool(
            data["reference_annotations_are_algorithmic"]
        ),
        "intentionally_open": True,
    }
    half = repair_trimesh(
        TriMesh(vertices, data["faces"].astype(int), metadata=metadata),
        intentionally_open=True,
    )
    return mirror_half_surface(half, axis=1)
