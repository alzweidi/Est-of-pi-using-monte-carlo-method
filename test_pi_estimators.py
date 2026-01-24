"""
Test suite for the pi estimators module.
Validates sampling methods, confidence intervals, and basic accuracy.
"""

import math

import numpy as np

from pi_estimators import SUPPORTED_METHODS, estimate_pi, generate_points, halton_sequence


def test_generate_points_lengths():
    rng = np.random.default_rng(123)
    n = 1234
    for method in SUPPORTED_METHODS:
        x, y = generate_points(method, n, rng)
        assert len(x) == n, f"{method} x length should be {n}, got {len(x)}"
        assert len(y) == n, f"{method} y length should be {n}, got {len(y)}"
    print("[PASS] test_generate_points_lengths")


def test_halton_sequence_bounds():
    seq = halton_sequence(1000, 2)
    assert np.all(seq >= 0), "Halton sequence values should be >= 0"
    assert np.all(seq < 1), "Halton sequence values should be < 1"
    print("[PASS] test_halton_sequence_bounds")


def test_wilson_interval_contains_estimate():
    result = estimate_pi("plain", 10000, seed=42)
    assert result.ci_low <= result.estimate <= result.ci_high, "Estimate should fall within CI"
    print("[PASS] test_wilson_interval_contains_estimate")


def test_methods_reasonable_accuracy():
    for method in SUPPORTED_METHODS:
        result = estimate_pi(method, 50000, seed=42)
        assert abs(result.estimate - math.pi) < 0.1, f"{method} estimate too far from pi"
    print("[PASS] test_methods_reasonable_accuracy")


def run_all_tests():
    print("=" * 60)
    print("PI ESTIMATORS - TEST SUITE")
    print("=" * 60)

    tests = [
        test_generate_points_lengths,
        test_halton_sequence_bounds,
        test_wilson_interval_contains_estimate,
        test_methods_reasonable_accuracy,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as exc:
            print(f"[FAIL] {test.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"[ERR]  {test.__name__}: {exc}")
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    raise SystemExit(0 if success else 1)
