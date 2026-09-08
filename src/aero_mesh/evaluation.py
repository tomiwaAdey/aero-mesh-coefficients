from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntervalEstimate:
    mean: float
    lower: float
    upper: float
    samples: int


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    iterations: int = 10_000,
    seed: int = 13,
) -> IntervalEstimate:
    """Return a percentile bootstrap interval over independent geometry cases."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return IntervalEstimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(iterations, len(array)))
    bootstrapped = np.mean(array[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    return IntervalEstimate(
        mean=float(np.mean(array)),
        lower=float(np.quantile(bootstrapped, tail)),
        upper=float(np.quantile(bootstrapped, 1.0 - tail)),
        samples=int(len(array)),
    )


def paired_metric_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    higher_is_better: bool,
    seed: int = 13,
) -> dict[str, float | int | str]:
    """Summarise a paired geometry-level comparison without inflating edge counts."""
    baseline_array = np.asarray(baseline, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    valid = np.isfinite(baseline_array) & np.isfinite(candidate_array)
    baseline_array = baseline_array[valid]
    candidate_array = candidate_array[valid]
    raw_delta = candidate_array - baseline_array
    improvement = raw_delta if higher_is_better else -raw_delta
    interval = bootstrap_mean_interval(improvement, seed=seed)

    baseline_mean = float(np.mean(baseline_array)) if len(baseline_array) else float("nan")
    candidate_mean = float(np.mean(candidate_array)) if len(candidate_array) else float("nan")
    denominator = abs(baseline_mean)
    relative = 100.0 * interval.mean / denominator if denominator > 1e-12 else float("nan")
    return {
        "samples": interval.samples,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "raw_delta_candidate_minus_baseline": float(np.mean(raw_delta)) if len(raw_delta) else float("nan"),
        "improvement": interval.mean,
        "improvement_ci_low": interval.lower,
        "improvement_ci_high": interval.upper,
        "relative_improvement_pct": relative,
        "higher_is_better": str(higher_is_better).lower(),
        "standardized_paired_effect": float(np.mean(improvement) / np.std(improvement, ddof=1))
        if len(improvement) > 1 and np.std(improvement, ddof=1) > 1e-12
        else float("nan"),
    }


def hierarchical_paired_metric_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    dataset_seeds: np.ndarray,
    base_geometry_ids: np.ndarray,
    *,
    higher_is_better: bool,
    confidence: float = 0.95,
    iterations: int = 10_000,
    seed: int = 13,
) -> dict[str, float | int | str]:
    """Bootstrap paired metrics by dataset seed, then physical geometry."""
    baseline_array = np.asarray(baseline, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    seeds = np.asarray(dataset_seeds)
    geometry_ids = np.asarray(base_geometry_ids)
    valid = np.isfinite(baseline_array) & np.isfinite(candidate_array)
    baseline_array = baseline_array[valid]
    candidate_array = candidate_array[valid]
    seeds = seeds[valid]
    geometry_ids = geometry_ids[valid]
    if not len(baseline_array):
        return {
            "rows": 0,
            "dataset_seeds": 0,
            "base_geometries": 0,
            "seed_geometry_pairs": 0,
            "baseline_mean": float("nan"),
            "candidate_mean": float("nan"),
            "improvement": float("nan"),
            "improvement_ci_low": float("nan"),
            "improvement_ci_high": float("nan"),
            "standardized_paired_effect": float("nan"),
            "higher_is_better": str(higher_is_better).lower(),
        }
    direction = 1.0 if higher_is_better else -1.0
    delta = direction * (candidate_array - baseline_array)
    unique_seeds = np.unique(seeds)
    seed_clusters = _seed_geometry_means(delta, seeds, geometry_ids, unique_seeds)
    baseline_clusters = _seed_geometry_means(
        baseline_array, seeds, geometry_ids, unique_seeds
    )
    candidate_clusters = _seed_geometry_means(
        candidate_array, seeds, geometry_ids, unique_seeds
    )
    samples = _hierarchical_bootstrap_samples(
        seed_clusters,
        iterations=iterations,
        seed=seed,
    )
    tail = (1.0 - confidence) / 2.0
    base_level_delta = np.concatenate(seed_clusters)
    seed_geometry_pairs = {
        (seed_value, geometry)
        for seed_value, geometry in zip(seeds.tolist(), geometry_ids.tolist())
    }
    effect_denominator = float(np.std(base_level_delta, ddof=1)) if len(base_level_delta) > 1 else 0.0
    baseline_mean = float(np.mean([np.mean(values) for values in baseline_clusters]))
    candidate_mean = float(np.mean([np.mean(values) for values in candidate_clusters]))
    improvement = direction * (candidate_mean - baseline_mean)
    return {
        "rows": int(len(delta)),
        "dataset_seeds": int(len(unique_seeds)),
        "base_geometries": int(len(np.unique(geometry_ids))),
        "seed_geometry_pairs": int(len(seed_geometry_pairs)),
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "improvement": improvement,
        "improvement_ci_low": float(np.quantile(samples, tail)),
        "improvement_ci_high": float(np.quantile(samples, 1.0 - tail)),
        "standardized_paired_effect": improvement / effect_denominator
        if effect_denominator > 1e-12
        else float("nan"),
        "higher_is_better": str(higher_is_better).lower(),
    }


def hierarchical_mean_interval(
    values: np.ndarray,
    dataset_seeds: np.ndarray,
    base_geometry_ids: np.ndarray,
    *,
    confidence: float = 0.95,
    iterations: int = 10_000,
    seed: int = 13,
) -> dict[str, float | int]:
    """Estimate a mean while preserving seed, geometry and remeshing hierarchy."""
    array = np.asarray(values, dtype=float)
    seeds = np.asarray(dataset_seeds)
    geometry_ids = np.asarray(base_geometry_ids)
    valid = np.isfinite(array)
    array = array[valid]
    seeds = seeds[valid]
    geometry_ids = geometry_ids[valid]
    if not len(array):
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "rows": 0,
            "dataset_seeds": 0,
            "base_geometries": 0,
            "seed_geometry_pairs": 0,
        }
    unique_seeds = np.unique(seeds)
    seed_geometry_pairs = {
        (seed_value, geometry)
        for seed_value, geometry in zip(seeds.tolist(), geometry_ids.tolist())
    }

    seed_clusters = _seed_geometry_means(array, seeds, geometry_ids, unique_seeds)
    point = float(np.mean([np.mean(values) for values in seed_clusters]))
    samples = _hierarchical_bootstrap_samples(
        seed_clusters,
        iterations=iterations,
        seed=seed,
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": point,
        "ci_low": float(np.quantile(samples, tail)),
        "ci_high": float(np.quantile(samples, 1.0 - tail)),
        "rows": int(len(array)),
        "dataset_seeds": int(len(unique_seeds)),
        "base_geometries": int(len(np.unique(geometry_ids))),
        "seed_geometry_pairs": int(len(seed_geometry_pairs)),
    }


def _seed_geometry_means(
    values: np.ndarray,
    seeds: np.ndarray,
    geometry_ids: np.ndarray,
    unique_seeds: np.ndarray,
) -> list[np.ndarray]:
    """Collapse remeshings while retaining one value per seed and physical geometry."""
    clusters: list[np.ndarray] = []
    for seed_value in unique_seeds:
        seed_mask = seeds == seed_value
        clusters.append(
            np.asarray(
                [
                    np.mean(values[seed_mask & (geometry_ids == geometry)])
                    for geometry in np.unique(geometry_ids[seed_mask])
                ],
                dtype=float,
            )
        )
    return clusters


def _hierarchical_bootstrap_samples(
    seed_clusters: list[np.ndarray],
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    """Resample dataset seeds first and physical geometries second.

    Each repeated draw of one dataset seed receives an independent geometry
    resample. The calculation is vectorised over bootstrap iterations; this is
    statistically equivalent to the direct nested loop but avoids millions of
    small Python indexing operations.
    """
    rng = np.random.default_rng(seed)
    cluster_count = len(seed_clusters)
    sampled_clusters = rng.integers(
        0,
        cluster_count,
        size=(iterations, cluster_count),
    )
    samples = np.zeros(iterations, dtype=float)
    for slot in range(cluster_count):
        chosen = sampled_clusters[:, slot]
        slot_means = np.empty(iterations, dtype=float)
        for cluster_index, values in enumerate(seed_clusters):
            mask = chosen == cluster_index
            draws = int(np.sum(mask))
            if not draws:
                continue
            indices = rng.integers(
                0,
                len(values),
                size=(draws, len(values)),
            )
            slot_means[mask] = np.mean(values[indices], axis=1)
        samples += slot_means / cluster_count
    return samples
