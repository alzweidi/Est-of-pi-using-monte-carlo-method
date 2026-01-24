"""Core estimation utilities for Monte Carlo pi experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


SUPPORTED_METHODS = ("plain", "stratified", "antithetic", "halton")


@dataclass(frozen=True)
class PiEstimate:
    method: str
    n: int
    estimate: float
    inside: int
    stderr: float
    ci_low: float
    ci_high: float
    runtime_s: float

    @property
    def error(self) -> float:
        return abs(self.estimate - math.pi)


@dataclass(frozen=True)
class TrialSummary:
    method: str
    n: int
    mean_estimate: float
    mean_error: float
    rmse: float
    std_dev: float
    mean_runtime_s: float
    ci_low: float
    ci_high: float
    variance_reduction: Optional[float]


@dataclass(frozen=True)
class ComparisonData:
    methods: List[str]
    sample_sizes: List[int]
    trials: int
    summaries: List[TrialSummary]


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    p_hat = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z2 / (4 * n * n))) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def generate_points(method: str, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample points for a given method."""
    if n <= 0:
        return np.array([]), np.array([])

    if method == "plain":
        x = rng.uniform(0, 1, n)
        y = rng.uniform(0, 1, n)
        return x, y

    if method == "antithetic":
        half = n // 2
        x_half = rng.uniform(0, 1, half)
        y_half = rng.uniform(0, 1, half)
        x = np.concatenate([x_half, 1 - x_half])
        y = np.concatenate([y_half, 1 - y_half])
        if n % 2 == 1:
            x = np.concatenate([x, rng.uniform(0, 1, 1)])
            y = np.concatenate([y, rng.uniform(0, 1, 1)])
        return x, y

    if method == "stratified":
        m = int(math.sqrt(n))
        if m <= 0:
            return np.array([]), np.array([])
        u = rng.uniform(0, 1, (m, m))
        v = rng.uniform(0, 1, (m, m))
        grid_x, grid_y = np.meshgrid(np.arange(m), np.arange(m), indexing="xy")
        x = (grid_x + u) / m
        y = (grid_y + v) / m
        x = x.ravel()
        y = y.ravel()
        remaining = n - (m * m)
        if remaining > 0:
            x_extra = rng.uniform(0, 1, remaining)
            y_extra = rng.uniform(0, 1, remaining)
            x = np.concatenate([x, x_extra])
            y = np.concatenate([y, y_extra])
        return x, y

    if method == "halton":
        x = halton_sequence(n, 2)
        y = halton_sequence(n, 3)
        shift = rng.random(2)
        x = (x + shift[0]) % 1.0
        y = (y + shift[1]) % 1.0
        return x, y

    raise ValueError(f"Unknown method '{method}'. Choose from {', '.join(SUPPORTED_METHODS)}.")


def halton_sequence(n: int, base: int) -> np.ndarray:
    """Generate n points of the Halton sequence for a given base."""
    if n <= 0:
        return np.array([])
    result = np.zeros(n)
    indices = np.arange(1, n + 1)
    f = 1.0
    while np.any(indices > 0):
        f /= base
        result += f * (indices % base)
        indices //= base
    return result


def estimate_pi(method: str, n: int, seed: Optional[int] = None, z: float = 1.96) -> PiEstimate:
    """Estimate pi using the selected sampling method."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {', '.join(SUPPORTED_METHODS)}.")

    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    x, y = generate_points(method, n, rng)
    inside = (x * x + y * y) <= 1.0
    inside_count = int(np.sum(inside))
    runtime_s = time.perf_counter() - start

    if n <= 0:
        return PiEstimate(method, n, 0.0, 0, 0.0, 0.0, 0.0, runtime_s)

    p_hat = inside_count / n
    estimate = 4 * p_hat
    stderr = 4 * math.sqrt(p_hat * (1 - p_hat) / n)
    ci_low_p, ci_high_p = wilson_interval(inside_count, n, z=z)
    return PiEstimate(
        method=method,
        n=n,
        estimate=estimate,
        inside=inside_count,
        stderr=stderr,
        ci_low=4 * ci_low_p,
        ci_high=4 * ci_high_p,
        runtime_s=runtime_s,
    )


def run_trials(method: str, n: int, trials: int, seed: Optional[int] = None) -> List[PiEstimate]:
    """Run repeated trials and return per-trial estimates."""
    if trials <= 0:
        return []
    if seed is None:
        return [estimate_pi(method, n, seed=None) for _ in range(trials)]
    seed_rng = np.random.default_rng(seed)
    seeds = seed_rng.integers(0, 2**32 - 1, size=trials, dtype=np.uint64)
    return [estimate_pi(method, n, seed=int(s)) for s in seeds]


def compare_methods(
    methods: Iterable[str],
    sample_sizes: Iterable[int],
    trials: int,
    seed: Optional[int] = None,
    z: float = 1.96,
) -> ComparisonData:
    """Compare methods across sample sizes and return summary statistics."""
    methods_list = [m for m in methods]
    sample_list = [int(n) for n in sample_sizes]
    summaries: List[TrialSummary] = []
    variance_map: Dict[Tuple[str, int], float] = {}

    if seed is None:
        seed_rng = None
    else:
        seed_rng = np.random.default_rng(seed)

    for method in methods_list:
        for n in sample_list:
            run_seed = None
            if seed_rng is not None:
                run_seed = int(seed_rng.integers(0, 2**32 - 1))
            results = run_trials(method, n, trials, seed=run_seed)
            estimates = np.array([r.estimate for r in results], dtype=float)
            runtimes = np.array([r.runtime_s for r in results], dtype=float)
            errors = np.abs(estimates - math.pi)

            mean_est = float(np.mean(estimates)) if estimates.size else 0.0
            mean_err = float(np.mean(errors)) if errors.size else 0.0
            rmse = float(math.sqrt(np.mean((estimates - math.pi) ** 2))) if estimates.size else 0.0
            std_dev = float(np.std(estimates, ddof=1)) if estimates.size > 1 else 0.0
            mean_runtime = float(np.mean(runtimes)) if runtimes.size else 0.0
            variance_map[(method, n)] = std_dev * std_dev

            if trials > 1:
                margin = z * std_dev / math.sqrt(trials)
                ci_low = mean_est - margin
                ci_high = mean_est + margin
            else:
                ci_low = mean_est
                ci_high = mean_est

            summaries.append(
                TrialSummary(
                    method=method,
                    n=n,
                    mean_estimate=mean_est,
                    mean_error=mean_err,
                    rmse=rmse,
                    std_dev=std_dev,
                    mean_runtime_s=mean_runtime,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    variance_reduction=None,
                )
            )

    # Compute variance reduction vs plain, when available.
    plain_variance: Dict[int, float] = {}
    for n in sample_list:
        variance = variance_map.get(("plain", n))
        if variance is not None:
            plain_variance[n] = variance

    updated_summaries: List[TrialSummary] = []
    for summary in summaries:
        baseline = plain_variance.get(summary.n)
        ratio = None
        if baseline is not None and summary.std_dev > 0:
            ratio = baseline / (summary.std_dev * summary.std_dev)
        updated_summaries.append(
            TrialSummary(
                method=summary.method,
                n=summary.n,
                mean_estimate=summary.mean_estimate,
                mean_error=summary.mean_error,
                rmse=summary.rmse,
                std_dev=summary.std_dev,
                mean_runtime_s=summary.mean_runtime_s,
                ci_low=summary.ci_low,
                ci_high=summary.ci_high,
                variance_reduction=ratio,
            )
        )

    return ComparisonData(
        methods=methods_list,
        sample_sizes=sample_list,
        trials=trials,
        summaries=updated_summaries,
    )


def summaries_by_method(data: ComparisonData) -> Dict[str, List[TrialSummary]]:
    """Group summary results by method."""
    grouped: Dict[str, List[TrialSummary]] = {m: [] for m in data.methods}
    for summary in data.summaries:
        grouped.setdefault(summary.method, []).append(summary)
    for method in grouped:
        grouped[method] = sorted(grouped[method], key=lambda s: s.n)
    return grouped
