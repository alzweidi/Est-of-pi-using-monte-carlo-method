# Monte Carlo Estimation of π

A Monte Carlo simulation to estimate π using random sampling and geometric probability.

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

## Available Scripts

| Script | Description |
|--------|-------------|
| `monte_carlo_pi.py` | Basic simulation with analysis report |
| `monte_carlo_pi_animated.py` | Publication-quality figures + real-time animation |
| `test_monte_carlo.py` | Test suite to verify correctness |
| `stat_analysis.py` | Convergence diagnostics, filtering, and confidence analysis |


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

Convergence and Statistical Analysis

In addition to a basic Monte Carlo estimator, this project implements convergence diagnostics commonly used in simulation-based inference.

An initial warmup phase is used to reduce early instability. After warmup, π estimates are filtered using a rolling z-score criterion to reduce the influence of extreme outliers.

A running mean of accepted estimates is tracked to visualize convergence behavior. To quantify uncertainty, a shrinking 95% confidence envelope is computed using the standard error of the running estimator:

```

SE(π̂ₙ) = sₙ / √n
```
The confidence band contracts at the expected O(n⁻¹ᐟ²) rate and is used to infer π after convergence rather than relying on a single terminal value.
Example Result

In a representative run, the estimator converged to:
```
π ≈ 3.14158

```
Notes

Statistical diagnostics are currently implemented in stat_analysis.py and will be integrated into a unified script in a future update.

Figures generated using Matplotlib are suitable for reports or publication-quality visualization.

## Example Output

<img width="1470" height="883" alt="Sample_fig_π_est" src="https://github.com/user-attachments/assets/374081ad-0076-4b60-9bac-27857caf4787" />

Example run showing the filtered running mean, shrinking 95% confidence envelope,
and convergence toward the true value of π.




Why This Project

This project demonstrates probabilistic modeling, Monte Carlo simulation, convergence diagnostics, and uncertainty quantification — core techniques used in quantitative finance, computational physics, and statistical modeling.

