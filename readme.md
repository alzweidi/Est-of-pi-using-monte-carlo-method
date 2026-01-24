# Monte Carlo Estimation of π

A Monte Carlo simulation to estimate π using random sampling and geometric probability.

## Highlights

- Multiple estimators: plain, stratified, antithetic, and low-discrepancy (Halton) sampling
- Confidence intervals, error metrics, and variance reduction comparisons
- CLI for one-off simulations, method comparisons, and report bundles
- Publication-quality visualizations and real-time animation

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Est-of-pi-using-monte-carlo-method.git

# 2. Navigate to the project
cd Est-of-pi-using-monte-carlo-method

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the simulation
python3 monte_carlo_pi.py
```

### New CLI (recommended)

```bash
# Single run with confidence interval
python3 pi_lab.py simulate --method stratified --samples 200000 --seed 42

# Compare methods across sample sizes
python3 pi_lab.py compare --methods plain stratified antithetic halton --samples 1000 10000 100000 --trials 5 --plot comparison.png

# Generate a full report bundle
python3 pi_lab.py report --out-dir reports
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `monte_carlo_pi.py` | Basic simulation with analysis report |
| `monte_carlo_pi_animated.py` | Publication-quality figures + real-time animation |
| `pi_lab.py` | CLI for simulation, comparison, and reporting |
| `test_monte_carlo.py` | Test suite to verify correctness |
| `test_pi_estimators.py` | Test suite for the estimation module |

### Run with Animation

```bash
python3 monte_carlo_pi_animated.py
```

Choose from:
1. **Static figure** — publication-quality PNG (300 DPI)
2. **Real-time animation** — watch the simulation run live
3. **Both**

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

## How It Works

By uniformly generating points inside a unit square and measuring the fraction that fall within a quarter circle, the algorithm approximates π:

```
π ≈ 4 × (points inside circle) / (total points)
```

The project demonstrates probabilistic modelling, simulation-based estimation, and convergence analysis — core techniques used in quantitative finance and computational physics.

## Reports

The `pi_lab.py report` command generates:

- `reports/comparison.png` — error and throughput comparison plot
- `reports/report.json` — structured results for reuse
- `reports/report.md` — Markdown table of results
