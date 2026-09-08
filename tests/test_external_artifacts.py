from pathlib import Path

import numpy as np

from aero_mesh.external import load_crm_lifting_surface_artifact, load_onera_m6_artifact
from aero_mesh.reconstruction import reconstruct_aero_mesh_from_edges
from aero_mesh.stl import canonicalize_aircraft_mesh, edge_feature_table, topology_metrics


ROOT = Path(__file__).resolve().parents[1]


def _canonical_labels(mesh):
    canonical, _ = canonicalize_aircraft_mesh(mesh)
    edges, features, labels = edge_feature_table(canonical)
    assert labels is not None
    return canonical, edges, features, labels


def test_onera_artifact_is_closed_oriented_and_has_immutable_boundary_annotations():
    mesh = load_onera_m6_artifact(ROOT / "vendor/onera-m6/onera-m6-primary.npz")
    topology = topology_metrics(mesh)
    assert topology["closed"] == 1
    assert topology["nonmanifold_edges"] == 0
    assert topology["consistently_oriented"] == 1
    assert mesh.metadata["reference_annotation_method"] == "preserved construction curves"
    assert not mesh.metadata["reference_annotations_are_algorithmic"]
    canonical, edges, _, labels = _canonical_labels(mesh)
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    assert reconstruct_aero_mesh_from_edges(canonical, edges, labels).valid


def test_crm_artifact_is_explicitly_open_oriented_and_has_immutable_boundary_annotations():
    mesh = load_crm_lifting_surface_artifact(
        ROOT / "vendor/nasa-crm-high-speed/dpw4-crm-wing-labelled.npz"
    )
    topology = topology_metrics(mesh)
    assert topology["closed"] == 0
    assert mesh.metadata["intentionally_open"]
    assert topology["nonmanifold_edges"] == 0
    assert topology["consistently_oriented"] == 1
    assert mesh.metadata["reference_annotations_are_algorithmic"]
    assert "minimum-cost paths" in mesh.metadata["reference_annotation_method"]
    canonical, edges, _, labels = _canonical_labels(mesh)
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    assert reconstruct_aero_mesh_from_edges(canonical, edges, labels).valid
