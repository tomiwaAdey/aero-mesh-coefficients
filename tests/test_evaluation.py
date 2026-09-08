import numpy as np

from aero_mesh.evaluation import (
    bootstrap_mean_interval,
    hierarchical_mean_interval,
    hierarchical_paired_metric_summary,
    paired_metric_summary,
)


def test_bootstrap_mean_interval_contains_observed_mean():
    estimate = bootstrap_mean_interval(np.array([1.0, 2.0, 3.0, 4.0]), iterations=2_000)
    assert estimate.mean == 2.5
    assert estimate.lower < estimate.mean < estimate.upper
    assert estimate.samples == 4


def test_paired_summary_normalises_lower_is_better_as_positive_improvement():
    summary = paired_metric_summary(
        np.array([0.3, 0.2, 0.4]),
        np.array([0.2, 0.1, 0.3]),
        higher_is_better=False,
    )
    assert summary["improvement"] > 0.0
    assert summary["raw_delta_candidate_minus_baseline"] < 0.0


def test_hierarchical_summary_resamples_seeds_and_base_geometries():
    baseline = np.array([0.5, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65, 0.55])
    candidate = baseline + 0.1
    seeds = np.array([13, 13, 13, 13, 29, 29, 29, 29])
    geometries = np.array(["a", "a", "b", "b", "a", "a", "b", "b"])
    summary = hierarchical_paired_metric_summary(
        baseline,
        candidate,
        seeds,
        geometries,
        higher_is_better=True,
        iterations=1_000,
    )
    assert np.isclose(summary["improvement"], 0.1)
    assert summary["dataset_seeds"] == 2
    assert summary["base_geometries"] == 2
    assert summary["seed_geometry_pairs"] == 4
    assert summary["rows"] == 8


def test_hierarchical_paired_means_collapse_remeshings_and_weight_seeds_equally():
    baseline = np.array([0.0, 2.0, 10.0, 0.0, 4.0, 8.0])
    candidate = np.array([1.0, 3.0, 12.0, 4.0, 8.0, 12.0])
    seeds = np.array([13, 13, 13, 29, 29, 29])
    geometries = np.array(["a", "a", "b", "a", "a", "a"])
    summary = hierarchical_paired_metric_summary(
        baseline,
        candidate,
        seeds,
        geometries,
        higher_is_better=True,
        iterations=1_000,
    )
    # Seed 13 averages geometry means 1 and 10; seed 29 has one geometry mean 4.
    assert np.isclose(summary["baseline_mean"], 4.75)
    assert np.isclose(summary["candidate_mean"], 7.5)
    assert np.isclose(summary["improvement"], 2.75)


def test_hierarchical_mean_weights_seeds_and_base_geometries_equally():
    values = np.array([1.0, 3.0, 10.0, 2.0, 4.0, 6.0, 8.0])
    seeds = np.array([13, 13, 13, 29, 29, 29, 29])
    geometries = np.array(["a", "a", "b", "a", "b", "b", "b"])
    summary = hierarchical_mean_interval(
        values,
        seeds,
        geometries,
        iterations=1_000,
    )
    # Seed 13: mean(mean(1, 3), 10) = 6. Seed 29: mean(2, mean(4, 6, 8)) = 4.
    assert np.isclose(summary["mean"], 5.0)
    assert summary["dataset_seeds"] == 2
    assert summary["base_geometries"] == 2
    assert summary["seed_geometry_pairs"] == 4
    assert summary["rows"] == 7
