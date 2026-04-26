"""Tests for vetinari.analytics.isolation_forest.

Covers:
- IsolationForest scores known outliers higher than inliers
- Uniform data produces scores near 0.5
- Warmup period: no scores returned until enough data collected
- PrimaryAnomalyDetector and HardwareAnomalyDetector interfaces
- Thread safety (concurrent observe calls)
"""

from __future__ import annotations

import random
import threading
from typing import Any

import pytest

from tests.factories import make_point_cluster as _make_cluster
from vetinari.analytics.isolation_forest import (
    HardwareAnomalyDetector,
    IsolationForest,
    IsolationTreeNode,
    PrimaryAnomalyDetector,
    _c,
)

# ---------------------------------------------------------------------------
# _c (average path length correction)
# ---------------------------------------------------------------------------


def test_c_trivial_cases() -> None:
    assert _c(0) == 0.0
    assert _c(1) == 0.0
    assert _c(2) == 1.0


def test_c_increases_with_n() -> None:
    """c(n) should be monotonically increasing for n >= 2."""
    prev = _c(2)
    for n in range(3, 20):
        cur = _c(n)
        assert cur > prev, f"c({n}) should be > c({n - 1})"
        prev = cur


# ---------------------------------------------------------------------------
# IsolationTreeNode
# ---------------------------------------------------------------------------


def test_isolation_tree_node_leaf_defaults() -> None:
    node = IsolationTreeNode()
    assert node.feature_idx == -1
    assert node.size == 0
    assert node.left is None
    assert node.right is None


def test_isolation_tree_node_frozen() -> None:
    node = IsolationTreeNode(feature_idx=0, split_value=1.5, size=10)
    with pytest.raises(AttributeError):
        node.feature_idx = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# IsolationForest.fit() validation
# ---------------------------------------------------------------------------


def test_isolation_forest_fit_empty_raises() -> None:
    forest = IsolationForest(n_trees=10, seed=0)
    with pytest.raises(ValueError, match="empty"):
        forest.fit([])


def test_isolation_forest_fit_empty_features_raises() -> None:
    forest = IsolationForest(n_trees=10, seed=0)
    with pytest.raises(ValueError, match="dimension"):
        forest.fit([[]])


def test_isolation_forest_score_before_fit_raises() -> None:
    forest = IsolationForest(n_trees=10, seed=0)
    with pytest.raises(RuntimeError, match="not been fitted"):
        forest.score([1.0, 2.0])


# ---------------------------------------------------------------------------
# IsolationForest scoring
# ---------------------------------------------------------------------------


def test_outlier_scores_higher_than_inlier() -> None:
    """A point far from the training cluster should score higher than inliers.

    We train on a tight cluster around (0.5, 0.5) and compare a normal
    point (0.5, 0.5) against a clear outlier (100.0, 100.0).
    """
    rng = random.Random(42)
    # Tight cluster: 500 points near (0.5, 0.5) with spread 0.1
    data = _make_cluster(500, [0.5, 0.5], 0.1, rng)

    forest = IsolationForest(n_trees=100, subsample_size=256, seed=42)
    forest.fit(data)

    inlier_score = forest.score([0.5, 0.5])
    outlier_score = forest.score([100.0, 100.0])

    assert outlier_score > inlier_score, (
        f"Outlier score {outlier_score:.4f} should exceed inlier score {inlier_score:.4f}"
    )


def test_outlier_classified_as_anomaly() -> None:
    """is_anomaly should return True for a far-out-of-distribution point."""
    rng = random.Random(7)
    data = _make_cluster(500, [0.0, 0.0, 0.0], 0.05, rng)
    forest = IsolationForest(n_trees=100, subsample_size=256, contamination=0.05, seed=7)
    forest.fit(data)

    assert forest.is_anomaly([50.0, 50.0, 50.0])


def test_inlier_scores_lower_than_outlier() -> None:
    """The mean inlier score should be clearly below the mean outlier score.

    We verify relative ordering rather than absolute threshold crossing
    because the IF threshold is distribution-dependent: with a tight cluster
    inlier paths are short and scores sit near 0.5, while outlier paths are
    even shorter (isolated immediately), so outlier scores are clearly higher.
    """
    rng = random.Random(99)
    data = _make_cluster(500, [0.0, 0.0, 0.0], 0.05, rng)
    forest = IsolationForest(n_trees=100, subsample_size=256, contamination=0.05, seed=99)
    forest.fit(data)

    inlier_scores = [forest.score([rng.uniform(-0.05, 0.05) for _ in range(3)]) for _ in range(50)]
    outlier_scores = [forest.score([rng.uniform(50.0, 100.0) for _ in range(3)]) for _ in range(50)]

    mean_inlier = sum(inlier_scores) / len(inlier_scores)
    mean_outlier = sum(outlier_scores) / len(outlier_scores)

    assert mean_outlier > mean_inlier, (
        f"Mean outlier score {mean_outlier:.4f} should exceed mean inlier score {mean_inlier:.4f}"
    )


def test_scores_in_unit_interval() -> None:
    """All scores must lie in [0, 1]."""
    rng = random.Random(1)
    data = _make_cluster(300, [5.0, 5.0], 2.0, rng)
    forest = IsolationForest(n_trees=50, subsample_size=128, seed=1)
    forest.fit(data)

    test_points = [[rng.uniform(-50.0, 50.0), rng.uniform(-50.0, 50.0)] for _ in range(100)]
    for pt in test_points:
        s = forest.score(pt)
        assert 0.0 <= s <= 1.0, f"Score {s} out of [0, 1] for point {pt}"


def test_uniform_data_scores_near_half() -> None:
    """With uniformly random training data every point has equal isolation
    depth, so scores should cluster near 0.5."""
    rng = random.Random(3)
    # Uniform data in [0, 1]^2
    data = [[rng.random(), rng.random()] for _ in range(500)]
    forest = IsolationForest(n_trees=100, subsample_size=256, seed=3)
    forest.fit(data)

    # Test 50 points drawn from the same uniform distribution
    test_pts = [[rng.random(), rng.random()] for _ in range(50)]
    scores = [forest.score(pt) for pt in test_pts]
    mean_score = sum(scores) / len(scores)

    # Mean should be reasonably close to 0.5 (allow ±0.2 tolerance)
    assert 0.3 <= mean_score <= 0.7, f"Uniform data mean score {mean_score:.3f} not near 0.5"


def test_refit_does_not_break_scoring() -> None:
    """Calling fit() twice must not raise and scoring still returns [0, 1]."""
    rng = random.Random(11)
    data1 = _make_cluster(200, [1.0, 1.0], 0.2, rng)
    data2 = _make_cluster(200, [2.0, 2.0], 0.2, rng)

    forest = IsolationForest(n_trees=50, seed=11)
    forest.fit(data1)
    forest.fit(data2)

    score = forest.score([2.0, 2.0])
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# PrimaryAnomalyDetector
# ---------------------------------------------------------------------------


def test_primary_detector_warmup_returns_none() -> None:
    """Scores should be None until warmup_samples observations have arrived."""
    det = PrimaryAnomalyDetector(warmup_samples=50, n_trees=20, seed=0)
    scores: list[Any] = []
    for _ in range(49):
        scores.append(det.observe(latency=1.0, error_rate=0.01))

    assert all(s is None for s in scores), "All pre-warmup scores must be None"


def test_primary_detector_scores_after_warmup() -> None:
    """After warmup_samples observations the detector must return float scores."""
    det = PrimaryAnomalyDetector(warmup_samples=50, n_trees=20, seed=0)
    rng = random.Random(5)

    for _ in range(50):
        det.observe(
            latency=rng.gauss(200, 10),
            error_rate=rng.gauss(0.01, 0.002),
            token_usage=rng.gauss(500, 20),
            quality_score=rng.gauss(0.9, 0.02),
        )

    # Next observation should return a score
    score = det.observe(latency=200.0, error_rate=0.01)
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_primary_detector_outlier_scores_high() -> None:
    """An extreme outlier must score higher than a normal observation."""
    det = PrimaryAnomalyDetector(warmup_samples=100, n_trees=50, seed=42)
    rng = random.Random(42)

    # Build up history with normal data
    for _ in range(100):
        det.observe(
            latency=rng.gauss(200, 10),
            error_rate=rng.gauss(0.01, 0.001),
            token_usage=rng.gauss(500, 20),
            quality_score=rng.gauss(0.9, 0.01),
            context_window_util=rng.gauss(0.5, 0.05),
            inspector_rejection_rate=rng.gauss(0.05, 0.01),
            output_length_trend=rng.gauss(0.0, 0.1),
            retry_rate=rng.gauss(0.02, 0.005),
        )

    normal_score = det.observe(
        latency=200.0,
        error_rate=0.01,
        token_usage=500.0,
        quality_score=0.9,
        context_window_util=0.5,
        inspector_rejection_rate=0.05,
        output_length_trend=0.0,
        retry_rate=0.02,
    )
    outlier_score = det.observe(
        latency=9999.0,
        error_rate=0.99,
        token_usage=99999.0,
        quality_score=0.01,
        context_window_util=1.0,
        inspector_rejection_rate=0.99,
        output_length_trend=100.0,
        retry_rate=1.0,
    )

    assert normal_score is not None
    assert outlier_score is not None
    assert outlier_score > normal_score, (
        f"Outlier score {outlier_score:.4f} should exceed normal score {normal_score:.4f}"
    )


def test_primary_detector_feature_names() -> None:
    """FEATURE_NAMES must contain all 8 expected dimensions."""
    expected = {
        "latency",
        "error_rate",
        "token_usage",
        "quality_score",
        "context_window_util",
        "inspector_rejection_rate",
        "output_length_trend",
        "retry_rate",
    }
    assert set(PrimaryAnomalyDetector.FEATURE_NAMES) == expected
    assert len(PrimaryAnomalyDetector.FEATURE_NAMES) == 8


def test_primary_detector_missing_features_default_to_zero() -> None:
    """Calling observe() with no kwargs should not raise."""
    det = PrimaryAnomalyDetector(warmup_samples=10, n_trees=5, seed=0)
    for _ in range(10):
        det.observe()  # all features default to 0.0
    # After warmup this should return a score without error
    score = det.observe()
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# HardwareAnomalyDetector
# ---------------------------------------------------------------------------


def test_hardware_detector_warmup_returns_none() -> None:
    det = HardwareAnomalyDetector(warmup_samples=30, n_trees=10, seed=0)
    scores: list[Any] = []
    for _ in range(29):
        scores.append(det.observe(gpu_util_pct=50.0, vram_util_pct=60.0))

    assert all(s is None for s in scores)


def test_hardware_detector_scores_after_warmup() -> None:
    det = HardwareAnomalyDetector(warmup_samples=30, n_trees=20, seed=1)
    rng = random.Random(1)

    for _ in range(30):
        det.observe(
            gpu_util_pct=rng.gauss(70, 5),
            vram_util_pct=rng.gauss(60, 4),
            model_load_unload_freq=rng.gauss(2.0, 0.3),
            cache_hit_rate=rng.gauss(0.85, 0.05),
        )

    score = det.observe(gpu_util_pct=70.0, vram_util_pct=60.0)
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_hardware_detector_feature_names() -> None:
    expected = {
        "gpu_util_pct",
        "vram_util_pct",
        "model_load_unload_freq",
        "cache_hit_rate",
    }
    assert set(HardwareAnomalyDetector.FEATURE_NAMES) == expected
    assert len(HardwareAnomalyDetector.FEATURE_NAMES) == 4


def test_hardware_detector_outlier_scores_high() -> None:
    """A GPU thrash event (100% util + frequent loads) must score high."""
    det = HardwareAnomalyDetector(warmup_samples=50, n_trees=50, seed=77)
    rng = random.Random(77)

    for _ in range(50):
        det.observe(
            gpu_util_pct=rng.gauss(65, 5),
            vram_util_pct=rng.gauss(55, 4),
            model_load_unload_freq=rng.gauss(1.0, 0.2),
            cache_hit_rate=rng.gauss(0.9, 0.03),
        )

    normal_score = det.observe(
        gpu_util_pct=65.0,
        vram_util_pct=55.0,
        model_load_unload_freq=1.0,
        cache_hit_rate=0.9,
    )
    # Thrash event: GPU pegged at 100%, VRAM full, very frequent swaps, cache cold
    thrash_score = det.observe(
        gpu_util_pct=100.0,
        vram_util_pct=99.5,
        model_load_unload_freq=50.0,
        cache_hit_rate=0.02,
    )

    assert normal_score is not None
    assert thrash_score is not None
    assert thrash_score > normal_score


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_primary_detector_thread_safe() -> None:
    """Concurrent calls to observe() must not raise or produce corrupt state."""
    det = PrimaryAnomalyDetector(warmup_samples=100, n_trees=20, seed=9)
    errors: list[Exception] = []

    def worker(worker_rng: random.Random) -> None:
        try:
            for _ in range(60):
                det.observe(
                    latency=worker_rng.gauss(200, 10),
                    error_rate=worker_rng.gauss(0.01, 0.001),
                )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(random.Random(i),)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"


def test_hardware_detector_thread_safe() -> None:
    """Concurrent hardware observations must not corrupt state."""
    det = HardwareAnomalyDetector(warmup_samples=50, n_trees=20, seed=13)
    errors: list[Exception] = []

    def worker(worker_rng: random.Random) -> None:
        try:
            for _ in range(40):
                det.observe(
                    gpu_util_pct=worker_rng.gauss(70, 5),
                    vram_util_pct=worker_rng.gauss(60, 4),
                )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(random.Random(i * 7),)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"


def test_isolation_forest_thread_safe_fit_score() -> None:
    """Simultaneous fit and score calls on the same forest must not crash."""
    rng = random.Random(21)
    data = _make_cluster(300, [0.5, 0.5], 0.2, rng)
    forest = IsolationForest(n_trees=50, subsample_size=128, seed=21)
    forest.fit(data)
    errors: list[Exception] = []

    def scorer(local_rng: random.Random) -> None:
        try:
            for _ in range(20):
                forest.score([local_rng.random(), local_rng.random()])
        except Exception as exc:
            errors.append(exc)

    def refitter(local_rng: random.Random) -> None:
        try:
            for _ in range(3):
                new_data = _make_cluster(100, [0.5, 0.5], 0.2, local_rng)
                forest.fit(new_data)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=scorer, args=(random.Random(i),)) for i in range(4)] + [
        threading.Thread(target=refitter, args=(random.Random(i + 100),)) for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
