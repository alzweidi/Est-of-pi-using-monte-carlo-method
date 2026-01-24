"""Command line interface for Monte Carlo pi experiments."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt

from pi_estimators import ComparisonData, SUPPORTED_METHODS, compare_methods, estimate_pi, summaries_by_method


PLOT_COLORS = {
    "plain": "#1f77b4",
    "stratified": "#2ca02c",
    "antithetic": "#ff7f0e",
    "halton": "#9467bd",
}


def parse_samples(values: Sequence[str]) -> List[int]:
    """Parse sample sizes from comma or space-delimited values."""
    items: List[str] = []
    for value in values:
        items.extend([part for part in value.split(",") if part.strip()])
    result: List[int] = []
    for item in items:
        cleaned = item.strip().replace("_", "")
        if not cleaned:
            continue
        if "e" in cleaned.lower():
            value = int(float(cleaned))
        else:
            value = int(cleaned)
        if value <= 0:
            raise argparse.ArgumentTypeError("Sample sizes must be positive integers.")
        result.append(value)
    if not result:
        raise argparse.ArgumentTypeError("At least one sample size is required.")
    return result


def parse_methods(values: Sequence[str]) -> List[str]:
    items: List[str] = []
    for value in values:
        items.extend([part for part in value.split(",") if part.strip()])
    methods = [item.strip().lower() for item in items if item.strip()]
    if not methods:
        raise argparse.ArgumentTypeError("At least one method is required.")
    unknown = [m for m in methods if m not in SUPPORTED_METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown method(s): {', '.join(unknown)}. Choose from {', '.join(SUPPORTED_METHODS)}."
        )
    return methods


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    sep = "-+-".join("-" * width for width in widths)
    body = [" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows]
    return "\n".join([line, sep] + body)


def format_comparison_rows(data: ComparisonData) -> List[List[str]]:
    rows: List[List[str]] = []
    for summary in sorted(data.summaries, key=lambda s: (s.n, s.method)):
        throughput = summary.n / summary.mean_runtime_s if summary.mean_runtime_s > 0 else 0.0
        var_red = f"{summary.variance_reduction:.2f}" if summary.variance_reduction else "n/a"
        rows.append(
            [
                summary.method,
                f"{summary.n:,}",
                f"{summary.mean_estimate:.8f}",
                f"{summary.mean_error:.6f}",
                f"{summary.rmse:.6f}",
                f"{summary.std_dev:.6f}",
                f"{summary.ci_low:.8f}",
                f"{summary.ci_high:.8f}",
                f"{summary.mean_runtime_s:.4f}",
                f"{throughput:,.0f}",
                var_red,
            ]
        )
    return rows


def plot_comparison(data: ComparisonData, save_path: Path) -> None:
    grouped = summaries_by_method(data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax_err, ax_speed = axes
    for method, summaries in grouped.items():
        if not summaries:
            continue
        n_vals = [s.n for s in summaries]
        rmse_vals = [s.rmse for s in summaries]
        throughput = [
            (s.n / s.mean_runtime_s) if s.mean_runtime_s > 0 else float("nan") for s in summaries
        ]
        color = PLOT_COLORS.get(method, "#333333")
        ax_err.plot(n_vals, rmse_vals, marker="o", label=method, color=color)
        ax_speed.plot(n_vals, throughput, marker="o", label=method, color=color)

    ax_err.set_xscale("log")
    ax_err.set_yscale("log")
    ax_err.set_title("Error (RMSE) vs Samples")
    ax_err.set_xlabel("Samples (log)")
    ax_err.set_ylabel("RMSE (log)")
    ax_err.grid(True, alpha=0.3, linestyle="--")
    ax_err.legend()

    ax_speed.set_xscale("log")
    ax_speed.set_yscale("log")
    ax_speed.set_title("Throughput vs Samples")
    ax_speed.set_xlabel("Samples (log)")
    ax_speed.set_ylabel("Samples per second (log)")
    ax_speed.grid(True, alpha=0.3, linestyle="--")
    ax_speed.legend()

    fig.suptitle("Monte Carlo pi Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, facecolor="white")
    plt.close(fig)


def write_json_report(data: ComparisonData, path: Path) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methods": data.methods,
        "sample_sizes": data.sample_sizes,
        "trials": data.trials,
        "results": [
            {
                "method": s.method,
                "n": s.n,
                "mean_estimate": s.mean_estimate,
                "mean_error": s.mean_error,
                "rmse": s.rmse,
                "std_dev": s.std_dev,
                "ci_low": s.ci_low,
                "ci_high": s.ci_high,
                "mean_runtime_s": s.mean_runtime_s,
                "variance_reduction": s.variance_reduction,
            }
            for s in data.summaries
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown_report(data: ComparisonData, path: Path) -> None:
    headers = [
        "Method",
        "Samples",
        "Mean Est",
        "Mean Error",
        "RMSE",
        "Std Dev",
        "CI Low",
        "CI High",
        "Runtime (s)",
        "Var Red",
    ]
    rows: List[List[str]] = []
    for summary in sorted(data.summaries, key=lambda s: (s.n, s.method)):
        rows.append(
            [
                summary.method,
                f"{summary.n:,}",
                f"{summary.mean_estimate:.8f}",
                f"{summary.mean_error:.6f}",
                f"{summary.rmse:.6f}",
                f"{summary.std_dev:.6f}",
                f"{summary.ci_low:.8f}",
                f"{summary.ci_high:.8f}",
                f"{summary.mean_runtime_s:.4f}",
                f"{summary.variance_reduction:.2f}" if summary.variance_reduction else "n/a",
            ]
        )

    lines = ["# Monte Carlo pi report", "", f"- Generated: {datetime.now(timezone.utc).isoformat()}"]
    lines.append(f"- Methods: {', '.join(data.methods)}")
    lines.append(f"- Samples: {', '.join(str(n) for n in data.sample_sizes)}")
    lines.append(f"- Trials: {data.trials}")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_simulate(args: argparse.Namespace) -> None:
    result = estimate_pi(args.method, args.samples, seed=args.seed, z=args.z)
    error = abs(result.estimate - math.pi)
    throughput = result.n / result.runtime_s if result.runtime_s > 0 else 0.0
    headers = ["Method", "Samples", "Estimate", "Error", "Std Err", "CI Low", "CI High", "Runtime (s)", "Throughput"]
    rows = [
        [
            result.method,
            f"{result.n:,}",
            f"{result.estimate:.10f}",
            f"{error:.8f}",
            f"{result.stderr:.8f}",
            f"{result.ci_low:.10f}",
            f"{result.ci_high:.10f}",
            f"{result.runtime_s:.4f}",
            f"{throughput:,.0f}",
        ]
    ]
    print(format_table(headers, rows))


def run_compare(args: argparse.Namespace) -> None:
    methods = parse_methods(args.methods)
    samples = parse_samples(args.samples)
    data = compare_methods(methods, samples, trials=args.trials, seed=args.seed, z=args.z)
    headers = [
        "Method",
        "Samples",
        "Mean Est",
        "Mean Error",
        "RMSE",
        "Std Dev",
        "CI Low",
        "CI High",
        "Runtime (s)",
        "Throughput",
        "Var Red",
    ]
    rows = format_comparison_rows(data)
    print(format_table(headers, rows))

    if args.plot:
        plot_comparison(data, Path(args.plot))
        print(f"\nSaved comparison plot to {args.plot}")

    if args.json_out:
        write_json_report(data, Path(args.json_out))
        print(f"Saved JSON report to {args.json_out}")

    if args.md_out:
        write_markdown_report(data, Path(args.md_out))
        print(f"Saved Markdown report to {args.md_out}")


def run_report(args: argparse.Namespace) -> None:
    methods = parse_methods(args.methods)
    samples = parse_samples(args.samples)
    out_dir = Path(args.out_dir)
    data = compare_methods(methods, samples, trials=args.trials, seed=args.seed, z=args.z)

    plot_path = out_dir / "comparison.png"
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"

    plot_comparison(data, plot_path)
    write_json_report(data, json_path)
    write_markdown_report(data, md_path)

    print(f"Saved comparison plot to {plot_path}")
    print(f"Saved JSON report to {json_path}")
    print(f"Saved Markdown report to {md_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monte Carlo pi lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Run a single simulation")
    simulate.add_argument("--method", default="plain", choices=SUPPORTED_METHODS)
    simulate.add_argument("--samples", type=int, default=100000)
    simulate.add_argument("--seed", type=int, default=None)
    simulate.add_argument("--z", type=float, default=1.96, help="Z-score for CI")
    simulate.set_defaults(func=run_simulate)

    compare = subparsers.add_parser("compare", help="Compare methods across sample sizes")
    compare.add_argument("--methods", nargs="+", default=list(SUPPORTED_METHODS))
    compare.add_argument("--samples", nargs="+", default=["1000", "10000", "100000"])
    compare.add_argument("--trials", type=int, default=5)
    compare.add_argument("--seed", type=int, default=None)
    compare.add_argument("--z", type=float, default=1.96, help="Z-score for CI")
    compare.add_argument("--plot", type=str, default=None, help="Path to save comparison plot")
    compare.add_argument("--json-out", type=str, default=None, help="Path to save JSON report")
    compare.add_argument("--md-out", type=str, default=None, help="Path to save Markdown report")
    compare.set_defaults(func=run_compare)

    report = subparsers.add_parser("report", help="Generate a full report bundle")
    report.add_argument("--methods", nargs="+", default=list(SUPPORTED_METHODS))
    report.add_argument("--samples", nargs="+", default=["2000", "20000", "200000"])
    report.add_argument("--trials", type=int, default=5)
    report.add_argument("--seed", type=int, default=42)
    report.add_argument("--z", type=float, default=1.96, help="Z-score for CI")
    report.add_argument("--out-dir", type=str, default="reports")
    report.set_defaults(func=run_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
