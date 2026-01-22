"""
Test suite for Monte Carlo π estimation.
Verifies mathematical correctness, convergence properties, and colour coding.
"""

import numpy as np
import sys

from monte_carlo_pi import estimate_pi, compute_convergence, get_accuracy_colour


def test_estimate_pi_returns_correct_types():
    """Verify estimate_pi returns correct types."""
    pi_est, x, y, inside = estimate_pi(100, seed=42)
    
    assert isinstance(pi_est, float), f"pi_est should be float, got {type(pi_est)}"
    assert isinstance(x, np.ndarray), f"x should be ndarray, got {type(x)}"
    assert isinstance(y, np.ndarray), f"y should be ndarray, got {type(y)}"
    assert isinstance(inside, np.ndarray), f"inside should be ndarray, got {type(inside)}"
    assert len(x) == 100, f"x length should be 100, got {len(x)}"
    assert len(y) == 100, f"y length should be 100, got {len(y)}"
    assert len(inside) == 100, f"inside length should be 100, got {len(inside)}"
    print("[PASS] test_estimate_pi_returns_correct_types")


def test_points_in_unit_square():
    """Verify all points are within [0, 1] x [0, 1]."""
    _, x, y, _ = estimate_pi(10000, seed=42)
    
    assert np.all(x >= 0) and np.all(x <= 1), "x values should be in [0, 1]"
    assert np.all(y >= 0) and np.all(y <= 1), "y values should be in [0, 1]"
    print("[PASS] test_points_in_unit_square")


def test_inside_classification_correct():
    """Verify points are correctly classified as inside/outside quarter circle."""
    _, x, y, inside = estimate_pi(10000, seed=42)
    
    distance_sq = x**2 + y**2
    expected_inside = distance_sq <= 1
    
    assert np.array_equal(inside, expected_inside), "Inside classification mismatch"
    print("[PASS] test_inside_classification_correct")


def test_pi_formula_correct():
    """Verify π = 4 * (inside count / total count)."""
    n = 10000
    pi_est, _, _, inside = estimate_pi(n, seed=42)
    
    expected_pi = 4 * np.sum(inside) / n
    assert abs(pi_est - expected_pi) < 1e-10, f"Formula mismatch: {pi_est} != {expected_pi}"
    print("[PASS] test_pi_formula_correct")


def test_reproducibility():
    """Verify same seed produces same results."""
    pi1, x1, y1, inside1 = estimate_pi(1000, seed=123)
    pi2, x2, y2, inside2 = estimate_pi(1000, seed=123)
    
    assert pi1 == pi2, "Same seed should produce same π estimate"
    assert np.array_equal(x1, x2), "Same seed should produce same x values"
    assert np.array_equal(y1, y2), "Same seed should produce same y values"
    print("[PASS] test_reproducibility")


def test_pi_estimate_reasonable():
    """Verify π estimate is in reasonable range with large sample."""
    pi_est, _, _, _ = estimate_pi(1_000_000, seed=42)
    
    assert 3.0 < pi_est < 3.3, f"π estimate {pi_est} is unreasonable"
    assert abs(pi_est - np.pi) < 0.01, f"With 1M samples, error should be < 0.01, got {abs(pi_est - np.pi)}"
    print(f"[PASS] test_pi_estimate_reasonable (pi = {pi_est:.6f}, error = {abs(pi_est - np.pi):.6f})")


def test_convergence_improves():
    """Verify error generally decreases with more samples."""
    checkpoints, estimates, errors = compute_convergence(100000, n_checkpoints=100, seed=42)
    
    early_avg_error = np.mean(errors[:10])
    late_avg_error = np.mean(errors[-10:])
    
    assert late_avg_error < early_avg_error, f"Error should decrease: early={early_avg_error:.4f}, late={late_avg_error:.4f}"
    print(f"[PASS] test_convergence_improves (early error: {early_avg_error:.4f}, late error: {late_avg_error:.4f})")


def test_colour_accuracy_thresholds():
    """Verify colour coding matches documented thresholds."""
    assert get_accuracy_colour(0.005) == '#00ff00', "Error < 0.01 should be green"
    assert get_accuracy_colour(0.03) == '#7fff00', "Error < 0.05 should be chartreuse"
    assert get_accuracy_colour(0.07) == '#ffff00', "Error < 0.10 should be yellow"
    assert get_accuracy_colour(0.15) == '#ff8c00', "Error < 0.20 should be orange"
    assert get_accuracy_colour(0.25) == '#ff0000', "Error >= 0.20 should be red"
    print("[PASS] test_colour_accuracy_thresholds")


def test_mathematical_basis():
    """
    Verify the mathematical basis:
    - Area of unit square = 1
    - Area of quarter circle (r=1) = π/4
    - Ratio = π/4, so π = 4 * ratio
    """
    n = 10_000_000
    np.random.seed(42)
    
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    inside = (x**2 + y**2) <= 1
    
    ratio = np.sum(inside) / n
    theoretical_ratio = np.pi / 4
    
    assert abs(ratio - theoretical_ratio) < 0.001, f"Ratio should be ~π/4: {ratio:.6f} vs {theoretical_ratio:.6f}"
    
    pi_from_ratio = 4 * ratio
    assert abs(pi_from_ratio - np.pi) < 0.005, f"π from ratio: {pi_from_ratio:.6f}"
    
    print(f"[PASS] test_mathematical_basis")
    print(f"   Ratio of points inside: {ratio:.6f} (theoretical π/4 = {theoretical_ratio:.6f})")
    print(f"   π = 4 × ratio = {pi_from_ratio:.10f}")
    print(f"   True π        = {np.pi:.10f}")
    print(f"   Error         = {abs(pi_from_ratio - np.pi):.10f}")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("MONTE CARLO π ESTIMATION - TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_estimate_pi_returns_correct_types,
        test_points_in_unit_square,
        test_inside_classification_correct,
        test_pi_formula_correct,
        test_reproducibility,
        test_pi_estimate_reasonable,
        test_convergence_improves,
        test_colour_accuracy_thresholds,
        test_mathematical_basis,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERR]  {test.__name__}: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
