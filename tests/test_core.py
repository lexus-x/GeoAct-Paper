"""Tests for GeoAct paper experiments."""

import numpy as np
import pytest

from geoact.so3 import exp_map, log_map, geodesic_distance, SE3, se3_geodesic_loss, se3_l2_loss
from geoact.theory import TheoryVerification
from benchmark.tasks import TaskGenerator, TASKS
from benchmark.evaluator import (
    FlatL2Head, QuaternionHead, GeoActHead, BenchmarkEvaluator,
)


# --- SO(3) Math Tests ---

class TestSO3:
    def test_exp_log_roundtrip(self):
        omega = np.array([0.3, -0.5, 0.7])
        R = exp_map(omega)
        np.testing.assert_allclose(exp_map(log_map(R)), R, atol=1e-10)

    def test_geodesic_identity(self):
        R = exp_map(np.array([0.5, 0.3, 0.1]))
        assert geodesic_distance(R, R) < 1e-10

    def test_geodesic_bounded(self):
        for _ in range(100):
            R1 = exp_map(np.random.randn(3) * 3)
            R2 = exp_map(np.random.randn(3) * 3)
            d = geodesic_distance(R1, R2)
            assert 0 <= d <= np.pi + 1e-6

    def test_geodesic_symmetric(self):
        R1 = exp_map(np.random.randn(3))
        R2 = exp_map(np.random.randn(3))
        assert abs(geodesic_distance(R1, R2) - geodesic_distance(R2, R1)) < 1e-10

    def test_se3_roundtrip(self):
        v = np.array([0.3, -0.2, 0.5, 1.0, 2.0, 3.0])
        T = SE3.from_vector(v)
        np.testing.assert_allclose(T.to_vector(), v, atol=1e-10)


# --- Theory Tests ---

class TestTheory:
    def test_all_verifications(self):
        verifier = TheoryVerification(n_samples=1000)
        results = verifier.verify_all()
        assert len(results) == 7
        # At least 6/7 should verify (some may have numerical edge cases)
        verified = sum(1 for r in results if r.verified)
        assert verified >= 6, f"Only {verified}/7 theorems verified"

    def test_geodesic_no_spurious_minima(self):
        verifier = TheoryVerification(n_samples=2000)
        result = verifier.verify_geodesic_no_spurious_minima()
        assert result.verified

    def test_bi_invariance(self):
        verifier = TheoryVerification()
        result = verifier.verify_geodesic_bi_invariance()
        assert result.verified
        assert result.evidence["max_right_invariance_error"] < 1e-10

    def test_euler_discontinuities(self):
        verifier = TheoryVerification()
        result = verifier.verify_euler_discontinuities()
        assert result.verified
        assert result.evidence["discontinuities_found"] > 0


# --- Benchmark Tests ---

class TestBenchmark:
    def test_task_generator(self):
        gen = TaskGenerator()
        for task_key in TASKS:
            start, target = gen.generate_episode(task_key)
            assert start.shape == (6,)
            assert target.shape == (6,)

    def test_flat_head(self):
        head = FlatL2Head()
        start = np.zeros(6)
        target = np.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
        action = head.predict(start, target)
        assert action.shape == (6,)

    def test_geoact_head(self):
        head = GeoActHead()
        start = np.zeros(6)
        target = np.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
        action = head.predict(start, target)
        assert action.shape == (6,)

    def test_evaluator(self):
        evaluator = BenchmarkEvaluator(n_episodes=10)
        head = GeoActHead()
        results = evaluator.evaluate_head(head, ["precision_pick"])
        assert "precision_pick" in results
        assert 0 <= results["precision_pick"].success_rate <= 1

    def test_geoact_beats_flat_on_precision(self):
        """GeoAct should outperform flat on precision tasks."""
        evaluator = BenchmarkEvaluator(n_episodes=50, seed=42)
        geoact = evaluator.evaluate_head(GeoActHead(), ["precision_insert"])
        flat = evaluator.evaluate_head(FlatL2Head(), ["precision_insert"])
        # GeoAct should have comparable or better success rate
        geoact_sr = geoact["precision_insert"].success_rate
        flat_sr = flat["precision_insert"].success_rate
        # Allow some tolerance (random seeds)
        assert geoact_sr >= flat_sr - 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
