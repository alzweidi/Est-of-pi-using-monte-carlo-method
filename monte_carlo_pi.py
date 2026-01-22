"""Monte Carlo Estimation of π using random sampling and geometric probability."""

from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def estimate_pi(n_points: int, seed: Optional[int] = None) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate π using Monte Carlo simulation.
    
    Args:
        n_points: Number of random points to generate
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Tuple of (pi_estimate, x_coords, y_coords, inside_mask)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    distance_squared = x**2 + y**2
    inside = distance_squared <= 1
    
    pi_estimate = 4 * np.sum(inside) / n_points
    
    return pi_estimate, x, y, inside


def compute_convergence(max_points: int, n_checkpoints: int = 500, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute π estimates at multiple sample sizes to analyze convergence.
    
    Args:
        max_points: Maximum number of points to simulate
        n_checkpoints: Number of checkpoints to record estimates
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (sample_sizes, pi_estimates, errors)
    """
    if seed is not None:
        np.random.seed(seed)
    
    checkpoints = np.unique(np.logspace(1, np.log10(max_points), n_checkpoints).astype(int))
    
    x = np.random.uniform(0, 1, max_points)
    y = np.random.uniform(0, 1, max_points)
    inside = (x**2 + y**2) <= 1
    
    cumulative_inside = np.cumsum(inside)
    
    pi_estimates = 4 * cumulative_inside[checkpoints - 1] / checkpoints
    errors = np.abs(pi_estimates - np.pi)
    
    return checkpoints, pi_estimates, errors


def get_accuracy_colour(error: float) -> str:
    """
    Return colour based on estimation error.
    Green = accurate, Yellow = moderate, Red = poor
    
    Args:
        error: Absolute error from true pi
    
    Returns:
        Hex colour string
    """
    if error < 0.01:
        return '#00ff00'  # Bright green - excellent
    elif error < 0.05:
        return '#7fff00'  # Chartreuse - very good
    elif error < 0.1:
        return '#ffff00'  # Yellow - good
    elif error < 0.2:
        return '#ff8c00'  # Dark orange - moderate
    else:
        return '#ff0000'  # Red - poor


def create_visualization(n_points: int = 10000, max_convergence_points: int = 100000, seed: int = 42) -> Tuple[float, float]:
    """
    Create a comprehensive visualization of the Monte Carlo π estimation.
    
    Includes:
    - Scatter plot of points (inside/outside quarter circle)
    - Real-time π estimate with color-coded accuracy
    - Convergence plot showing estimate approaching π
    - Error analysis plot showing 1/√n decay
    
    Args:
        n_points: Number of points for scatter visualization
        max_convergence_points: Maximum points for convergence analysis
        seed: Random seed for reproducibility
    """
    pi_est, x, y, inside = estimate_pi(n_points, seed=seed)
    checkpoints, pi_estimates, errors = compute_convergence(max_convergence_points, seed=seed + 1)
    
    fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')
    fig.suptitle('Monte Carlo Estimation of π', fontsize=24, fontweight='bold', color='white', y=0.98)
    
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1], 
                           hspace=0.3, wspace=0.25, left=0.08, right=0.95, top=0.9, bottom=0.08)
    
    # --- Plot 1: Scatter plot of points ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0f0f23')
    
    ax1.scatter(x[~inside], y[~inside], c='#ff6b6b', s=1, alpha=0.6, label='Outside')
    ax1.scatter(x[inside], y[inside], c='#4ecdc4', s=1, alpha=0.6, label='Inside')
    
    theta = np.linspace(0, np.pi/2, 100)
    arc_x = np.cos(theta)
    arc_y = np.sin(theta)
    ax1.plot(arc_x, arc_y, 'white', linewidth=2, label='Quarter Circle')
    
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x', fontsize=12, color='white')
    ax1.set_ylabel('y', fontsize=12, color='white')
    ax1.set_title(f'Random Points (n = {n_points:,})', fontsize=14, fontweight='bold', color='white')
    ax1.legend(loc='upper right', fontsize=10, facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('white')
    
    # --- Plot 2: π estimate display ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0f0f23')
    ax2.axis('off')
    
    error = abs(pi_est - np.pi)
    accuracy_colour = get_accuracy_colour(error)
    
    ax2.text(0.5, 0.85, 'Estimated π', fontsize=18, ha='center', va='center', 
             color='white', transform=ax2.transAxes, fontweight='bold')
    
    ax2.text(0.5, 0.6, f'{pi_est:.10f}', fontsize=36, ha='center', va='center',
             color=accuracy_colour, transform=ax2.transAxes, fontweight='bold',
             fontfamily='monospace')
    
    ax2.text(0.5, 0.4, 'True π', fontsize=18, ha='center', va='center',
             color='white', transform=ax2.transAxes, fontweight='bold')
    
    ax2.text(0.5, 0.25, f'{np.pi:.10f}', fontsize=28, ha='center', va='center',
             color='#888888', transform=ax2.transAxes, fontfamily='monospace')
    
    error_pct = (error / np.pi) * 100
    ax2.text(0.5, 0.08, f'Error: {error:.6f} ({error_pct:.4f}%)', fontsize=14, 
             ha='center', va='center', color=accuracy_colour, transform=ax2.transAxes)
    
    accuracy_labels = [
        ('< 0.01', '#00ff00', 'Excellent'),
        ('< 0.05', '#7fff00', 'Very Good'),
        ('< 0.10', '#ffff00', 'Good'),
        ('< 0.20', '#ff8c00', 'Moderate'),
        ('≥ 0.20', '#ff0000', 'Poor')
    ]
    
    for i, (threshold, color, label) in enumerate(accuracy_labels):
        y_pos = 0.95 - i * 0.08
        ax2.add_patch(plt.Rectangle((0.02, y_pos - 0.025), 0.04, 0.05, 
                                     facecolor=color, transform=ax2.transAxes))
        ax2.text(0.08, y_pos, f'{threshold}: {label}', fontsize=9, va='center',
                 color='white', transform=ax2.transAxes)
    
    # --- Plot 3: Convergence plot ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#0f0f23')
    
    colours = [get_accuracy_colour(e) for e in errors]
    
    for i in range(len(checkpoints) - 1):
        ax3.plot(checkpoints[i:i+2], pi_estimates[i:i+2], color=colours[i], linewidth=1.5, alpha=0.8)
    
    ax3.axhline(y=np.pi, color='white', linestyle='--', linewidth=2, label=f'True π = {np.pi:.6f}')
    
    ax3.fill_between(checkpoints, np.pi - 0.01, np.pi + 0.01, alpha=0.2, color='#00ff00', label='±0.01 band')
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Number of Samples (log scale)', fontsize=12, color='white')
    ax3.set_ylabel('π Estimate', fontsize=12, color='white')
    ax3.set_title('Convergence of π Estimate', fontsize=14, fontweight='bold', color='white')
    ax3.legend(loc='upper right', fontsize=10, facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax3.tick_params(colors='white')
    ax3.set_ylim(2.8, 3.5)
    for spine in ax3.spines.values():
        spine.set_color('white')
    ax3.grid(True, alpha=0.2, color='white')
    
    # --- Plot 4: Error analysis ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#0f0f23')
    
    ax4.scatter(checkpoints, errors, c=colours, s=10, alpha=0.7)
    
    theoretical_error = 1.0 / np.sqrt(checkpoints)
    ax4.plot(checkpoints, theoretical_error, 'white', linestyle='--', linewidth=2, 
             label=r'Theoretical: $O(1/\sqrt{n})$')
    
    ax4.axhline(y=0.01, color='#00ff00', linestyle=':', linewidth=1.5, alpha=0.7, label='Excellent threshold (0.01)')
    ax4.axhline(y=0.05, color='#7fff00', linestyle=':', linewidth=1.5, alpha=0.7, label='Very Good threshold (0.05)')
    
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Number of Samples (log scale)', fontsize=12, color='white')
    ax4.set_ylabel('Absolute Error (log scale)', fontsize=12, color='white')
    ax4.set_title('Error Analysis', fontsize=14, fontweight='bold', color='white')
    ax4.legend(loc='upper right', fontsize=9, facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax4.tick_params(colors='white')
    for spine in ax4.spines.values():
        spine.set_color('white')
    ax4.grid(True, alpha=0.2, color='white')
    
    plt.savefig('monte_carlo_pi_visualization.png', dpi=150, facecolor='#1a1a2e', edgecolor='none')
    plt.show()
    
    return pi_est, error


def run_analysis(sample_sizes: Optional[List[int]] = None, seed: int = 42) -> None:
    """
    Run Monte Carlo analysis for multiple sample sizes and print results.
    
    Args:
        sample_sizes: List of sample sizes to test
        seed: Random seed for reproducibility
    """
    if sample_sizes is None:
        sample_sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    
    print("=" * 70)
    print("MONTE CARLO π ESTIMATION - ANALYSIS REPORT")
    print("=" * 70)
    print(f"{'Samples':>12} | {'Estimate':>14} | {'Error':>12} | {'Error %':>10} | Status")
    print("-" * 70)
    
    for n in sample_sizes:
        pi_est, _, _, _ = estimate_pi(n, seed=seed)
        error = abs(pi_est - np.pi)
        error_pct = (error / np.pi) * 100
        
        if error < 0.01:
            status = "[OK] Excellent"
        elif error < 0.05:
            status = "[OK] Very Good"
        elif error < 0.1:
            status = "[--] Good"
        elif error < 0.2:
            status = "[??] Moderate"
        else:
            status = "[!!] Poor"
        
        print(f"{n:>12,} | {pi_est:>14.10f} | {error:>12.8f} | {error_pct:>9.5f}% | {status}")
    
    print("-" * 70)
    print(f"{'True π':>12} | {np.pi:>14.10f}")
    print("=" * 70)


if __name__ == "__main__":
    run_analysis()
    
    print("\nGenerating visualization...")
    pi_estimate, final_error = create_visualization(
        n_points=15000,
        max_convergence_points=500000,
        seed=42
    )
    
    print(f"\nFinal estimate: π ≈ {pi_estimate:.10f}")
    print(f"Final error: {final_error:.8f}")
    print("\nVisualization saved to 'monte_carlo_pi_visualization.png'")
