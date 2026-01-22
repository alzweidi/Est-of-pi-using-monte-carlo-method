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
