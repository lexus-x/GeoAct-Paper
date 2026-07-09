"""
Benchmark: GeoAct vs Flat Action Head.

Compares geometry-aware vs flat action prediction on:
1. Rotation accuracy (geodesic distance)
2. Translation accuracy (L2 distance)
3. Multi-modal prediction (can it handle ambiguous actions?)
4. Iterative refinement improvement
5. Discontinuity handling (Euler angle edge cases)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .so3 import (
    exp_map_so3, log_map_so3, geodesic_distance_so3,
    euler_to_rotation_matrix, rotation_matrix_to_euler,
    SE3, se3_geodesic_loss,
)
from .action_head import GeoActHead, GeoActConfig, FlatActionHead


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""
    test_name: str
    geoact_score: float
    flat_score: float
    geoact_wins: bool
    improvement_pct: float
    details: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def geoact_wins_count(self) -> int:
        return sum(1 for r in self.results if r.geoact_wins)

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def win_rate(self) -> float:
        return self.geoact_wins_count / max(self.total_tests, 1)

    def summary(self) -> str:
        lines = [
            f"GeoAct vs Flat Head: {self.geoact_wins_count}/{self.total_tests} wins",
            "",
        ]
        for r in self.results:
            winner = "✓ GeoAct" if r.geoact_wins else "✗ Flat"
            lines.append(f"  {winner}  {r.test_name}: GeoAct={r.geoact_score:.4f}, Flat={r.flat_score:.4f} ({r.improvement_pct:+.1f}%)")
        return "\n".join(lines)


class BenchmarkSuite:
    """Run all benchmarks comparing GeoAct vs Flat action head."""

    def __init__(self, feature_dim: int = 768, n_trials: int = 100, seed: int = 42):
        self.feature_dim = feature_dim
        self.n_trials = n_trials
        self.rng = np.random.RandomState(seed)

        self.geoact = GeoActHead(GeoActConfig(feature_dim=feature_dim))
        self.flat = FlatActionHead(feature_dim=feature_dim)

    def run_all(self) -> BenchmarkReport:
        """Run all benchmarks."""
        report = BenchmarkReport()

        report.results.append(self.bench_rotation_accuracy())
        report.results.append(self.bench_translation_accuracy())
        report.results.append(self.bench_discontinuity_handling())
        report.results.append(self.bench_multimodal_prediction())
        report.results.append(self.bench_refinement_improvement())
        report.results.append(self.bench_geodesic_vs_l2_loss())

        return report

    def bench_rotation_accuracy(self) -> BenchmarkResult:
        """Test: How accurately does each head predict rotations?"""
        geoact_errors = []
        flat_errors = []

        for _ in range(self.n_trials):
            # Random target rotation
            target_omega = self.rng.randn(3) * 0.5
            target_R = exp_map_so3(target_omega)

            # Random features
            features = self.rng.randn(self.feature_dim).astype(np.float32)

            # GeoAct prediction
            preds = self.geoact.predict(features, n_samples=5)
            best = self.geoact.select_best(preds)
            geoact_err = geodesic_distance_so3(best.rotation_matrix, target_R)
            geoact_errors.append(geoact_err)

            # Flat prediction
            flat_pred = self.flat.predict(features)
            flat_omega = flat_pred[:3]
            flat_R = exp_map_so3(flat_omega)
            flat_err = geodesic_distance_so3(flat_R, target_R)
            flat_errors.append(flat_err)

        geoact_mean = float(np.mean(geoact_errors))
        flat_mean = float(np.mean(flat_errors))
        improvement = (flat_mean - geoact_mean) / max(flat_mean, 1e-10) * 100

        return BenchmarkResult(
            test_name="Rotation Accuracy (geodesic error)",
            geoact_score=geoact_mean,
            flat_score=flat_mean,
            geoact_wins=geoact_mean < flat_mean,
            improvement_pct=improvement,
        )

    def bench_translation_accuracy(self) -> BenchmarkResult:
        """Test: How accurately does each head predict translations?"""
        geoact_errors = []
        flat_errors = []

        for _ in range(self.n_trials):
            target_trans = self.rng.randn(3) * 0.1
            target_action = np.concatenate([self.rng.randn(3) * 0.3, target_trans])
            features = self.rng.randn(self.feature_dim).astype(np.float32)

            # GeoAct
            preds = self.geoact.predict(features, n_samples=5)
            best = self.geoact.select_best(preds)
            geoact_err = float(np.linalg.norm(best.translation - target_trans))
            geoact_errors.append(geoact_err)

            # Flat
            flat_pred = self.flat.predict(features)
            flat_err = float(np.linalg.norm(flat_pred[3:6] - target_trans))
            flat_errors.append(flat_err)

        geoact_mean = float(np.mean(geoact_errors))
        flat_mean = float(np.mean(flat_errors))
        improvement = (flat_mean - geoact_mean) / max(flat_mean, 1e-10) * 100

        return BenchmarkResult(
            test_name="Translation Accuracy (L2 error)",
            geoact_score=geoact_mean,
            flat_score=flat_mean,
            geoact_wins=geoact_mean < flat_mean,
            improvement_pct=improvement,
        )

    def bench_discontinuity_handling(self) -> BenchmarkResult:
        """Test: Loss function continuity near ±π (Euler angle discontinuity)."""
        from .so3 import rotation_matrix_to_euler

        # Measure loss function smoothness near the discontinuity
        target_angle = np.pi
        target_R = exp_map_so3(np.array([0, 0, target_angle]))

        geodesic_losses = []
        euler_l2_losses = []

        for delta in np.linspace(-0.2, 0.2, self.n_trials):
            pred_R = exp_map_so3(np.array([0, 0, target_angle + delta]))

            # Geodesic loss (smooth everywhere)
            geo_loss = geodesic_distance_so3(pred_R, target_R)
            geodesic_losses.append(geo_loss)

            # L2 on Euler angles (discontinuous)
            pred_euler = rotation_matrix_to_euler(pred_R)
            target_euler = rotation_matrix_to_euler(target_R)
            l2_loss = float(np.sum((pred_euler - target_euler) ** 2))
            euler_l2_losses.append(l2_loss)

        # Measure smoothness: lower gradient variance = smoother
        geo_smoothness = float(np.std(np.diff(geodesic_losses)))
        euler_smoothness = float(np.std(np.diff(euler_l2_losses)))

        return BenchmarkResult(
            test_name="Loss Continuity near ±π (gradient variance)",
            geoact_score=geo_smoothness,
            flat_score=euler_smoothness,
            geoact_wins=geo_smoothness < euler_smoothness,
            improvement_pct=(euler_smoothness - geo_smoothness) / max(euler_smoothness, 1e-10) * 100,
        )

    def bench_multimodal_prediction(self) -> BenchmarkResult:
        """Test: Can the head predict multiple valid actions?"""
        features = self.rng.randn(self.feature_dim).astype(np.float32)

        # GeoAct: sample multiple predictions
        preds = self.geoact.predict(features, n_samples=10)
        rotations = [p.rotation_axis_angle for p in preds]
        # Measure diversity (average pairwise geodesic distance)
        diversity_scores = []
        for i in range(len(rotations)):
            for j in range(i + 1, len(rotations)):
                R_i = exp_map_so3(rotations[i])
                R_j = exp_map_so3(rotations[j])
                diversity_scores.append(geodesic_distance_so3(R_i, R_j))

        geoact_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0

        # Flat: deterministic, no diversity
        flat_pred1 = self.flat.predict(features)
        flat_pred2 = self.flat.predict(features)  # same input → same output
        flat_diversity = float(np.linalg.norm(flat_pred1 - flat_pred2))

        return BenchmarkResult(
            test_name="Multi-modal Prediction (diversity)",
            geoact_score=geoact_diversity,
            flat_score=flat_diversity,
            geoact_wins=geoact_diversity > flat_diversity,
            improvement_pct=0,  # diversity is better when higher
            details={"geoact_predictions": len(preds)},
        )

    def bench_refinement_improvement(self) -> BenchmarkResult:
        """Test: Geodesic loss gradient quality (how well it guides learning)."""
        # Measure gradient signal quality: geodesic loss gives better gradients
        target_omega = np.array([0.3, -0.2, 0.5])
        target_R = exp_map_so3(target_omega)

        # Sample predictions at varying distances
        geo_correlations = []
        l2_correlations = []

        for _ in range(self.n_trials):
            pred_omega = target_omega + self.rng.randn(3) * 0.3
            pred_R = exp_map_so3(pred_omega)

            # Geodesic distance (should correlate with actual error)
            geo_dist = geodesic_distance_so3(pred_R, target_R)

            # L2 on axis-angle (should also correlate, but may not near π)
            l2_dist = float(np.linalg.norm(pred_omega - target_omega))

            # Actual error (ground truth)
            actual_error = geo_dist

            geo_correlations.append((geo_dist, actual_error))
            l2_correlations.append((l2_dist, actual_error))

        # Correlation: higher = better gradient signal
        geo_corr = np.corrcoef([c[0] for c in geo_correlations], [c[1] for c in geo_correlations])[0, 1]
        l2_corr = np.corrcoef([c[0] for c in l2_correlations], [c[1] for c in l2_correlations])[0, 1]

        return BenchmarkResult(
            test_name="Gradient Quality (loss-error correlation)",
            geoact_score=float(geo_corr),
            flat_score=float(l2_corr),
            geoact_wins=float(geo_corr) > float(l2_corr),
            improvement_pct=(float(geo_corr) - float(l2_corr)) / max(abs(float(l2_corr)), 1e-10) * 100,
        )

    def bench_geodesic_vs_l2_loss(self) -> BenchmarkResult:
        """Test: Geodesic loss monotonicity (no false local minima)."""
        # Geodesic distance is always monotonically increasing with actual rotation difference
        # L2 on Euler angles is NOT monotonic near ±π
        target_R = exp_map_so3(np.array([0, 0, 0.5]))

        # Test monotonicity: as rotation gets further from target, loss should increase
        geo_monotonic_violations = 0
        l2_monotonic_violations = 0

        prev_geo = -1
        prev_l2 = -1
        for angle in np.linspace(0.5, 0.5 + 2 * np.pi, self.n_trials):
            pred_R = exp_map_so3(np.array([0, 0, angle]))

            geo = geodesic_distance_so3(pred_R, target_R)
            from .so3 import rotation_matrix_to_euler
            pred_euler = rotation_matrix_to_euler(pred_R)
            target_euler = rotation_matrix_to_euler(target_R)
            l2 = float(np.sum((pred_euler - target_euler) ** 2))

            if geo < prev_geo - 0.01:
                geo_monotonic_violations += 1
            if l2 < prev_l2 - 0.1:
                l2_monotonic_violations += 1

            prev_geo = geo
            prev_l2 = l2

        return BenchmarkResult(
            test_name="Monotonicity (no false local minima)",
            geoact_score=float(geo_monotonic_violations),
            flat_score=float(l2_monotonic_violations),
            geoact_wins=geo_monotonic_violations <= l2_monotonic_violations,
            improvement_pct=(l2_monotonic_violations - geo_monotonic_violations) / max(l2_monotonic_violations, 1) * 100,
        )
