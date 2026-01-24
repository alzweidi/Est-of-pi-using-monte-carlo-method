"""Sleek technical noir dashboard for Monte Carlo pi experiments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pi_estimators import SUPPORTED_METHODS, compare_methods, estimate_pi, generate_points, summaries_by_method


THEME = {
    "bg": "#070b10",
    "bg_alt": "#0b1118",
    "panel": "#121821",
    "panel_alt": "#0f141b",
    "border": "rgba(255,255,255,0.08)",
    "grid": "rgba(255,255,255,0.06)",
    "text": "#e6edf3",
    "muted": "#9aa4b2",
    "accent": "#00f5d4",
    "accent_alt": "#3aa3ff",
    "warning": "#ffb454",
}

METHOD_COLORS = {
    "plain": "#3aa3ff",
    "stratified": "#00f5d4",
    "antithetic": "#ffb454",
    "halton": "#7dd3fc",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        :root {
            --bg: #070b10;
            --bg-alt: #0b1118;
            --panel: #121821;
            --panel-alt: #0f141b;
            --border: rgba(255,255,255,0.08);
            --grid: rgba(255,255,255,0.06);
            --text: #e6edf3;
            --muted: #9aa4b2;
            --accent: #00f5d4;
            --accent-alt: #3aa3ff;
            --warning: #ffb454;
        }

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text);
        }

        .stApp {
            background:
                radial-gradient(900px 600px at 15% -10%, rgba(0, 245, 212, 0.10), transparent 65%),
                radial-gradient(700px 500px at 90% 0%, rgba(58, 163, 255, 0.12), transparent 55%),
                linear-gradient(180deg, #070b10 0%, #06090d 60%, #05070b 100%);
        }

        section[data-testid="stSidebar"] {
            background: var(--bg-alt);
            border-right: 1px solid var(--border);
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: var(--text);
        }

        .hero {
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            background: linear-gradient(135deg, rgba(18, 24, 33, 0.85), rgba(12, 17, 26, 0.7));
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
            position: relative;
            overflow: hidden;
        }

        .hero:after {
            content: "";
            position: absolute;
            inset: 0;
            background-image: linear-gradient(transparent 95%, rgba(255,255,255,0.05) 96%),
                              linear-gradient(90deg, transparent 95%, rgba(255,255,255,0.04) 96%);
            background-size: 22px 22px;
            opacity: 0.4;
            pointer-events: none;
        }

        .hero-title {
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
            color: var(--text);
        }

        .hero-sub {
            font-size: 1rem;
            color: var(--muted);
            max-width: 720px;
        }

        .signal-row {
            display: flex;
            gap: 0.6rem;
            margin-top: 0.9rem;
            flex-wrap: wrap;
        }

        .signal-pill {
            border: 1px solid var(--border);
            background: rgba(6, 10, 14, 0.6);
            color: var(--muted);
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            letter-spacing: 0.04em;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.35);
        }

        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
        }

        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 0.35rem;
            color: var(--accent);
        }

        .metric-sub {
            font-size: 0.8rem;
            color: var(--muted);
            margin-top: 0.3rem;
        }

        .panel {
            background: var(--panel-alt);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.1rem;
        }

        .panel-title {
            font-size: 0.9rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.6rem;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            margin-top: 0.5rem;
        }

        .data-table th, .data-table td {
            border-bottom: 1px solid var(--border);
            padding: 0.55rem 0.4rem;
            text-align: left;
        }

        .data-table th {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.75rem;
            color: var(--muted);
        }

        div.stButton > button {
            background: linear-gradient(135deg, rgba(0, 245, 212, 0.18), rgba(58, 163, 255, 0.2));
            border: 1px solid rgba(0, 245, 212, 0.4);
            color: var(--text);
            font-weight: 600;
            padding: 0.45rem 0.9rem;
            border-radius: 12px;
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            border-color: rgba(0, 245, 212, 0.7);
            box-shadow: 0 0 18px rgba(0, 245, 212, 0.25);
            transform: translateY(-1px);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: var(--panel-alt);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.35rem 0.9rem;
        }

        .stTabs [aria-selected="true"] {
            border-color: rgba(0, 245, 212, 0.5);
            box-shadow: inset 0 0 0 1px rgba(0, 245, 212, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_samples(value: int) -> str:
    return f"{value:,}"


def parse_sample_input(text: str) -> List[int]:
    items = [item.strip() for item in text.replace("\n", ",").split(",")]
    sizes: List[int] = []
    for item in items:
        if not item:
            continue
        cleaned = item.replace("_", "")
        if "e" in cleaned.lower():
            value = int(float(cleaned))
        else:
            value = int(cleaned)
        if value <= 0:
            raise ValueError("Sample sizes must be positive integers.")
        sizes.append(value)
    if not sizes:
        raise ValueError("Provide at least one sample size.")
    return sizes


def confidence_z(level: str) -> float:
    lookup = {"90%": 1.645, "95%": 1.96, "99%": 2.576}
    return lookup.get(level, 1.96)


@st.cache_data(show_spinner=False)
def cached_estimate(method: str, n: int, seed: int, z: float):
    return estimate_pi(method, n, seed=seed, z=z)


@st.cache_data(show_spinner=False)
def cached_points(method: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    return generate_points(method, n, rng)


@st.cache_data(show_spinner=False)
def cached_convergence(method: str, max_points: int, checkpoints: int, seed: int):
    rng = np.random.default_rng(seed)
    x, y = generate_points(method, max_points, rng)
    inside = (x * x + y * y) <= 1.0
    cumulative = np.cumsum(inside)
    idx = np.unique(np.logspace(1, math.log10(max_points), checkpoints).astype(int))
    estimates = 4 * cumulative[idx - 1] / idx
    return idx, estimates


def build_scatter(x: np.ndarray, y: np.ndarray, inside: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x[~inside],
            y=y[~inside],
            mode="markers",
            marker=dict(color="#ff6b6b", size=3, opacity=0.45),
            name="Outside",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=x[inside],
            y=y[inside],
            mode="markers",
            marker=dict(color=THEME["accent"], size=3, opacity=0.6),
            name="Inside",
        )
    )
    theta = np.linspace(0, math.pi / 2, 200)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line=dict(color=THEME["text"], width=2),
            name="Arc",
        )
    )
    fig.update_layout(
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["panel"],
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(range=[-0.02, 1.02], showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.02, 1.02], showgrid=False, zeroline=False, scaleanchor="x"),
        title="Sampled Points",
        font=dict(color=THEME["text"]),
    )
    return fig


def build_convergence_plot(checkpoints: np.ndarray, estimates: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=checkpoints,
            y=estimates,
            mode="lines",
            line=dict(color=THEME["accent_alt"], width=2),
            name="Estimate",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[checkpoints.min(), checkpoints.max()],
            y=[math.pi, math.pi],
            mode="lines",
            line=dict(color=THEME["warning"], width=1.5, dash="dash"),
            name="pi",
        )
    )
    fig.update_layout(
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["panel"],
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(type="log", title="Samples (log)", gridcolor=THEME["grid"]),
        yaxis=dict(title="Estimate", gridcolor=THEME["grid"]),
        title="Convergence",
        font=dict(color=THEME["text"]),
    )
    return fig


def render_metric(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_table(headers: List[str], rows: List[List[str]]) -> None:
    header_html = "".join([f"<th>{header}</th>" for header in headers])
    row_html = []
    for row in rows:
        row_html.append("<tr>" + "".join([f"<td>{cell}</td>" for cell in row]) + "</tr>")
    table_html = f"""
    <table class="data-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{"".join(row_html)}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def build_comparison_rows(data) -> List[List[str]]:
    rows: List[List[str]] = []
    for summary in sorted(data.summaries, key=lambda s: (s.n, s.method)):
        rows.append(
            [
                summary.method,
                format_samples(summary.n),
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
    return rows


def build_comparison_figures(data):
    grouped = summaries_by_method(data)
    rmse_fig = go.Figure()
    speed_fig = go.Figure()
    for method, summaries in grouped.items():
        if not summaries:
            continue
        n_vals = [s.n for s in summaries]
        rmse_vals = [s.rmse for s in summaries]
        throughput = [s.n / s.mean_runtime_s if s.mean_runtime_s > 0 else None for s in summaries]
        color = METHOD_COLORS.get(method, THEME["accent"])
        rmse_fig.add_trace(
            go.Scatter(
                x=n_vals,
                y=rmse_vals,
                mode="lines+markers",
                name=method,
                line=dict(color=color, width=2),
            )
        )
        speed_fig.add_trace(
            go.Scatter(
                x=n_vals,
                y=throughput,
                mode="lines+markers",
                name=method,
                line=dict(color=color, width=2),
            )
        )

    rmse_fig.update_layout(
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["panel"],
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(type="log", title="Samples (log)", gridcolor=THEME["grid"]),
        yaxis=dict(type="log", title="RMSE (log)", gridcolor=THEME["grid"]),
        title="RMSE vs Samples",
        font=dict(color=THEME["text"]),
    )
    speed_fig.update_layout(
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["panel"],
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(type="log", title="Samples (log)", gridcolor=THEME["grid"]),
        yaxis=dict(type="log", title="Samples per second (log)", gridcolor=THEME["grid"]),
        title="Throughput vs Samples",
        font=dict(color=THEME["text"]),
    )
    return rmse_fig, speed_fig


def build_json_report(data) -> str:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methods": data.methods,
        "sample_sizes": data.sample_sizes,
        "trials": data.trials,
        "results": [asdict(summary) for summary in data.summaries],
    }
    return json.dumps(payload, indent=2)


def build_markdown_report(data) -> str:
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
    rows = build_comparison_rows(data)
    lines = [
        "# Monte Carlo pi report",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Methods: {', '.join(data.methods)}",
        f"- Samples: {', '.join(str(n) for n in data.sample_sizes)}",
        f"- Trials: {data.trials}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def highlight_summary(data):
    if not data.summaries:
        return None
    top_n = max(data.sample_sizes)
    subset = [s for s in data.summaries if s.n == top_n]
    if not subset:
        return None
    best_rmse = min(subset, key=lambda s: s.rmse)
    fastest = max(
        subset,
        key=lambda s: (s.n / s.mean_runtime_s) if s.mean_runtime_s > 0 else 0,
    )
    return best_rmse, fastest


def main() -> None:
    st.set_page_config(page_title="Monte Carlo Pi Lab", layout="wide")
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Monte Carlo Pi Lab</div>
            <div class="hero-sub">
                Sleek technical noir interface for probing Monte Carlo estimators, confidence bands,
                and variance reduction with real-time visuals.
            </div>
            <div class="signal-row">
                <span class="signal-pill">plain sampling</span>
                <span class="signal-pill">stratified grid</span>
                <span class="signal-pill">antithetic pairs</span>
                <span class="signal-pill">halton sequence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    tabs = st.tabs(["Single Run", "Method Comparison", "About"])

    with st.sidebar:
        st.markdown("## Control Room")
        st.markdown("Tune the sampler and signal profile.", unsafe_allow_html=True)
        st.markdown("---")
        method = st.selectbox("Sampling method", list(SUPPORTED_METHODS))
        samples = st.number_input("Samples", min_value=1000, max_value=2_000_000, value=200000, step=1000)
        confidence = st.selectbox("Confidence level", ["90%", "95%", "99%"], index=1)
        use_random = st.checkbox("Random seed", value=False)
        seed_value = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
        z_value = confidence_z(confidence)

        st.markdown("---")
        st.markdown("### Visual Load")
        viz_points = st.slider("Scatter points", min_value=1000, max_value=15000, value=5000, step=500)
        convergence_points = st.slider(
            "Convergence max samples", min_value=20000, max_value=200000, value=100000, step=10000
        )
        convergence_checkpoints = st.slider(
            "Convergence checkpoints", min_value=50, max_value=300, value=160, step=10
        )
        run_single = st.button("Run simulation")

        st.markdown("---")
        st.markdown("### Comparison Batch")
        comp_methods = st.multiselect(
            "Methods",
            list(SUPPORTED_METHODS),
            default=list(SUPPORTED_METHODS),
        )
        comp_samples = st.text_input("Sample sizes", value="1000, 10000, 100000")
        comp_trials = st.number_input("Trials per method", min_value=1, max_value=20, value=5, step=1)
        run_compare = st.button("Run comparison")

    with tabs[0]:
        params = (method, samples, use_random, seed_value, z_value)
        if "single_params" not in st.session_state:
            st.session_state.single_params = params
        if "single_seed" not in st.session_state:
            st.session_state.single_seed = seed_value

        if run_single or st.session_state.single_params != params:
            st.session_state.single_params = params
            if use_random:
                st.session_state.single_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
            else:
                st.session_state.single_seed = seed_value

        active_seed = st.session_state.single_seed
        result = cached_estimate(method, samples, active_seed, z_value)
        throughput = result.n / result.runtime_s if result.runtime_s > 0 else 0.0
        error = abs(result.estimate - math.pi)

        metrics = st.columns(4, gap="large")
        with metrics[0]:
            render_metric("Estimate", f"{result.estimate:.10f}", f"Seed {active_seed}")
        with metrics[1]:
            render_metric("Absolute Error", f"{error:.6f}", f"True pi {math.pi:.6f}")
        with metrics[2]:
            render_metric(
                "Confidence Band",
                f"{result.ci_low:.6f} - {result.ci_high:.6f}",
                f"Std err {result.stderr:.6f}",
            )
        with metrics[3]:
            render_metric("Throughput", f"{throughput:,.0f}/s", f"Runtime {result.runtime_s:.4f}s")

        st.write("")
        chart_cols = st.columns(2, gap="large")

        x, y = cached_points(method, viz_points, active_seed)
        inside = (x * x + y * y) <= 1.0
        scatter_fig = build_scatter(x, y, inside)
        checkpoints, estimates = cached_convergence(method, convergence_points, convergence_checkpoints, active_seed + 7)
        convergence_fig = build_convergence_plot(checkpoints, estimates)

        with chart_cols[0]:
            st.plotly_chart(scatter_fig, use_container_width=True)
        with chart_cols[1]:
            st.plotly_chart(convergence_fig, use_container_width=True)

        st.write("")
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Signal Summary</div>
                <div>
                    Monte Carlo estimate pi = 4 * (inside / total). This run uses the selected sampler
                    and reports a Wilson confidence interval for the binomial ratio.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tabs[1]:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Batch Comparison</div>
                <div>
                    Compare variance reduction and throughput across multiple sampling methods.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        data = None
        if run_compare and comp_methods:
            try:
                sizes = parse_sample_input(comp_samples)
                with st.spinner("Running comparison..."):
                    data = compare_methods(comp_methods, sizes, trials=int(comp_trials), seed=42)
            except ValueError as exc:
                st.error(str(exc))

        if data:
            highlights = highlight_summary(data)
            if highlights:
                best_rmse, fastest = highlights
                summary_cols = st.columns(2, gap="large")
                with summary_cols[0]:
                    render_metric(
                        "Lowest RMSE",
                        f"{best_rmse.method} @ {format_samples(best_rmse.n)}",
                        f"RMSE {best_rmse.rmse:.6f}",
                    )
                with summary_cols[1]:
                    throughput = (
                        fastest.n / fastest.mean_runtime_s if fastest.mean_runtime_s > 0 else 0.0
                    )
                    render_metric(
                        "Fastest",
                        f"{fastest.method} @ {format_samples(fastest.n)}",
                        f"{throughput:,.0f}/s",
                    )

            st.write("")
            rmse_fig, speed_fig = build_comparison_figures(data)
            chart_cols = st.columns(2, gap="large")
            with chart_cols[0]:
                st.plotly_chart(rmse_fig, use_container_width=True)
            with chart_cols[1]:
                st.plotly_chart(speed_fig, use_container_width=True)

            st.write("")
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
            render_table(headers, build_comparison_rows(data))

            json_report = build_json_report(data)
            md_report = build_markdown_report(data)
            st.download_button(
                "Download JSON",
                data=json_report,
                file_name="pi_report.json",
                mime="application/json",
            )
            st.download_button(
                "Download Markdown",
                data=md_report,
                file_name="pi_report.md",
                mime="text/markdown",
            )

        if not run_compare:
            st.info("Run a comparison from the Control Room to populate this panel.")

    with tabs[2]:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">System Brief</div>
                <div>
                    The lab estimates pi by sampling points in a unit square and tracking the
                    proportion that fall inside the quarter circle. Variance reduction techniques
                    (stratified, antithetic, low-discrepancy) tighten the estimate faster than
                    plain Monte Carlo.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.code(
            "pi = 4 * (points_inside / total_points)\n"
            "stderr = 4 * sqrt(p_hat * (1 - p_hat) / n)",
            language="text",
        )


if __name__ == "__main__":
    main()
