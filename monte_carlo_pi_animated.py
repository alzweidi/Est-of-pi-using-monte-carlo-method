"""
Monte Carlo Estimation of pi - Publication Quality Visualisation with Real-Time Animation.

This module provides both static and animated visualisations suitable for
academic presentations and publications. Uses LaTeX rendering for proper
mathematical typography.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import matplotlib.patheffects as path_effects
from typing import Callable


# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,  # Set True if LaTeX is installed
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

# Professional colour palette (colourblind-friendly)
COLOURS = {
    'inside': '#2ecc71',      # Emerald green
    'outside': '#e74c3c',     # Alizarin red  
    'accent': '#3498db',      # Peter river blue
    'background': '#0d1117',  # Dark background
    'panel': '#161b22',       # Panel background
    'grid': '#30363d',        # Grid lines
    'text': '#e6edf3',        # Light text
    'text_dim': '#8b949e',    # Dimmed text
    'gold': '#f1c40f',        # Gold accent
    'theoretical': '#9b59b6', # Amethyst purple
}


def estimate_pi(n_points: int, seed: int | None = None) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate pi using Monte Carlo simulation."""
    if seed is not None:
        np.random.seed(seed)
    
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    distance_squared = x**2 + y**2
    inside = distance_squared <= 1
    
    pi_estimate = 4 * np.sum(inside) / n_points
    
    return pi_estimate, x, y, inside


def get_accuracy_colour(error: float) -> str:
    """Return colour based on estimation error."""
    if error < 0.01:
        return '#00d26a'  # Bright green
    elif error < 0.05:
        return '#88d317'  # Lime
    elif error < 0.1:
        return '#ffc300'  # Amber
    elif error < 0.2:
        return '#ff8c00'  # Orange
    else:
        return '#ff4757'  # Red


def create_publication_figure(
    n_points: int = 20000,
    max_convergence_points: int = 500000,
    seed: int = 42,
    save_path: str | None = 'monte_carlo_pi_publication.png'
) -> tuple[float, float]:
    """
    Create publication-quality static visualisation.
    
    Suitable for academic papers, presentations, and theses.
    """
    np.random.seed(seed)
    
    # Generate data
    pi_est, x, y, inside = estimate_pi(n_points, seed=seed)
    
    # Convergence data
    np.random.seed(seed + 1)
    max_n = max_convergence_points
    x_conv = np.random.uniform(0, 1, max_n)
    y_conv = np.random.uniform(0, 1, max_n)
    inside_conv = (x_conv**2 + y_conv**2) <= 1
    
    checkpoints = np.unique(np.logspace(1, np.log10(max_n), 400).astype(int))
    cumsum = np.cumsum(inside_conv)
    pi_estimates = 4 * cumsum[checkpoints - 1] / checkpoints
    errors = np.abs(pi_estimates - np.pi)
    
    # Create figure
    fig = plt.figure(figsize=(14, 10), facecolor=COLOURS['background'])
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.3, 1], width_ratios=[1.2, 0.8, 1],
                           hspace=0.35, wspace=0.3, left=0.07, right=0.96, top=0.92, bottom=0.08)
    
    # Main title
    fig.suptitle(r'Monte Carlo Estimation of $\pi$', fontsize=20, fontweight='bold',
                 color=COLOURS['text'], y=0.97)
    
    # --- Panel 1: Scatter plot ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLOURS['panel'])
    
    # Plot points with transparency gradient based on density
    ax1.scatter(x[~inside], y[~inside], c=COLOURS['outside'], s=0.8, alpha=0.5, 
                rasterized=True, label=f'Outside ({np.sum(~inside):,})')
    ax1.scatter(x[inside], y[inside], c=COLOURS['inside'], s=0.8, alpha=0.5,
                rasterized=True, label=f'Inside ({np.sum(inside):,})')
    
    # Quarter circle arc
    theta = np.linspace(0, np.pi/2, 200)
    ax1.plot(np.cos(theta), np.sin(theta), color=COLOURS['text'], linewidth=2, 
             label='Unit circle', zorder=10)
    
    # Boundary box
    ax1.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=COLOURS['text_dim'], 
             linewidth=1, linestyle='--', alpha=0.5)
    
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'$x$', color=COLOURS['text'])
    ax1.set_ylabel(r'$y$', color=COLOURS['text'])
    ax1.set_title(f'Random Sampling (n = {n_points:,})', fontweight='bold', 
                  color=COLOURS['text'], pad=10)
    ax1.legend(loc='upper right', facecolor=COLOURS['panel'], edgecolor=COLOURS['grid'],
               labelcolor=COLOURS['text'], framealpha=0.9)
    _style_axis(ax1)
    
    # --- Panel 2: Results display ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLOURS['panel'])
    ax2.axis('off')
    
    error = abs(pi_est - np.pi)
    accuracy_colour = get_accuracy_colour(error)
    
    # Estimated value
    ax2.text(0.5, 0.88, 'Estimated', fontsize=14, ha='center', va='center',
             color=COLOURS['text_dim'], transform=ax2.transAxes)
    ax2.text(0.5, 0.78, r'$\hat{\pi}$', fontsize=24, ha='center', va='center',
             color=COLOURS['text'], transform=ax2.transAxes)
    ax2.text(0.5, 0.62, f'{pi_est:.8f}', fontsize=22, ha='center', va='center',
             color=accuracy_colour, transform=ax2.transAxes, fontweight='bold',
             family='monospace')
    
    # True value
    ax2.text(0.5, 0.45, 'True Value', fontsize=12, ha='center', va='center',
             color=COLOURS['text_dim'], transform=ax2.transAxes)
    ax2.text(0.5, 0.35, r'$\pi$', fontsize=20, ha='center', va='center',
             color=COLOURS['text'], transform=ax2.transAxes)
    ax2.text(0.5, 0.22, f'{np.pi:.8f}', fontsize=18, ha='center', va='center',
             color=COLOURS['text_dim'], transform=ax2.transAxes, family='monospace')
    
    # Error metrics
    error_pct = (error / np.pi) * 100
    ax2.text(0.5, 0.08, f'Error: {error:.2e} ({error_pct:.4f}%)', fontsize=11,
             ha='center', va='center', color=accuracy_colour, transform=ax2.transAxes)
    
    # Border
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, 
                          edgecolor=COLOURS['grid'], linewidth=1, transform=ax2.transAxes)
    ax2.add_patch(rect)
    
    # --- Panel 3: Method explanation ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(COLOURS['panel'])
    ax3.axis('off')
    
    method_text = [
        (0.5, 0.92, 'Method', 14, 'bold', COLOURS['text']),
        (0.5, 0.78, r'Area ratio: $\frac{A_{circle}}{A_{square}} = \frac{\pi/4}{1}$', 12, 'normal', COLOURS['text']),
        (0.5, 0.62, r'$\pi = 4 \times \frac{N_{inside}}{N_{total}}$', 14, 'normal', COLOURS['gold']),
        (0.5, 0.45, 'Statistics', 14, 'bold', COLOURS['text']),
        (0.5, 0.32, f'Samples: {n_points:,}', 11, 'normal', COLOURS['text_dim']),
        (0.5, 0.22, f'Inside: {np.sum(inside):,} ({100*np.sum(inside)/n_points:.2f}%)', 11, 'normal', COLOURS['inside']),
        (0.5, 0.12, f'Outside: {np.sum(~inside):,} ({100*np.sum(~inside)/n_points:.2f}%)', 11, 'normal', COLOURS['outside']),
    ]
    
    for x_pos, y_pos, text, size, weight, colour in method_text:
        ax3.text(x_pos, y_pos, text, fontsize=size, ha='center', va='center',
                 color=colour, transform=ax3.transAxes, fontweight=weight)
    
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False,
                          edgecolor=COLOURS['grid'], linewidth=1, transform=ax3.transAxes)
    ax3.add_patch(rect)
    
    # --- Panel 4: Convergence plot ---
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor(COLOURS['panel'])
    
    # Colour gradient based on error
    for i in range(len(checkpoints) - 1):
        colour = get_accuracy_colour(errors[i])
        ax4.plot(checkpoints[i:i+2], pi_estimates[i:i+2], color=colour, linewidth=1.2, alpha=0.9)
    
    # True pi line
    ax4.axhline(y=np.pi, color=COLOURS['theoretical'], linestyle='-', linewidth=2,
                label=r'$\pi = 3.14159...$', zorder=5)
    
    # Confidence bands
    ax4.fill_between(checkpoints, np.pi - 0.01, np.pi + 0.01, alpha=0.15,
                     color=COLOURS['inside'], label=r'$\pm 0.01$ band')
    ax4.fill_between(checkpoints, np.pi - 0.001, np.pi + 0.001, alpha=0.25,
                     color=COLOURS['inside'])
    
    ax4.set_xscale('log')
    ax4.set_xlabel('Number of Samples (n)', color=COLOURS['text'])
    ax4.set_ylabel(r'$\hat{\pi}$ Estimate', color=COLOURS['text'])
    ax4.set_title('Convergence Analysis', fontweight='bold', color=COLOURS['text'], pad=10)
    ax4.set_ylim(2.9, 3.4)
    ax4.legend(loc='upper right', facecolor=COLOURS['panel'], edgecolor=COLOURS['grid'],
               labelcolor=COLOURS['text'], framealpha=0.9)
    _style_axis(ax4)
    ax4.grid(True, alpha=0.3, color=COLOURS['grid'], linestyle='-', linewidth=0.5)
    
    # --- Panel 5: Error decay ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(COLOURS['panel'])
    
    # Scatter with colour coding
    colours = [get_accuracy_colour(e) for e in errors]
    ax5.scatter(checkpoints, errors, c=colours, s=8, alpha=0.7, rasterized=True)
    
    # Theoretical O(1/sqrt(n)) line
    theoretical = 1.6 / np.sqrt(checkpoints)  # Scaled for visibility
    ax5.plot(checkpoints, theoretical, color=COLOURS['theoretical'], linestyle='--',
             linewidth=2, label=r'$O(1/\sqrt{n})$ theoretical')
    
    # Threshold lines
    ax5.axhline(y=0.01, color='#00d26a', linestyle=':', linewidth=1.5, alpha=0.7)
    ax5.axhline(y=0.001, color='#00d26a', linestyle=':', linewidth=1, alpha=0.5)
    
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel('Number of Samples (n)', color=COLOURS['text'])
    ax5.set_ylabel(r'$|\hat{\pi} - \pi|$', color=COLOURS['text'])
    ax5.set_title('Error Decay', fontweight='bold', color=COLOURS['text'], pad=10)
    ax5.legend(loc='upper right', facecolor=COLOURS['panel'], edgecolor=COLOURS['grid'],
               labelcolor=COLOURS['text'], framealpha=0.9, fontsize=8)
    _style_axis(ax5)
    ax5.grid(True, alpha=0.3, color=COLOURS['grid'], linestyle='-', linewidth=0.5)
    
    if save_path:
        plt.savefig(save_path, facecolor=COLOURS['background'], edgecolor='none',
                    bbox_inches='tight', dpi=300)
        print(f"Saved publication figure to '{save_path}'")
    
    plt.show()
    return pi_est, error


def run_realtime_animation(
    n_points: int = 5000,
    batch_size: int = 50,
    interval_ms: int = 50,
    seed: int | None = None
) -> None:
    """
    Run real-time animated Monte Carlo simulation.
    
    Watch the points appear and pi estimate converge in real time.
    
    Args:
        n_points: Total number of points to simulate
        batch_size: Points added per frame
        interval_ms: Milliseconds between frames
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Pre-generate all points for smooth animation
    all_x = np.random.uniform(0, 1, n_points)
    all_y = np.random.uniform(0, 1, n_points)
    all_inside = (all_x**2 + all_y**2) <= 1
    
    # Setup figure
    fig = plt.figure(figsize=(14, 7), facecolor=COLOURS['background'])
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.25,
                           left=0.06, right=0.97, top=0.88, bottom=0.1)
    
    fig.suptitle(r'Real-Time Monte Carlo Simulation of $\pi$', fontsize=18,
                 fontweight='bold', color=COLOURS['text'], y=0.96)
    
    # Left panel: scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLOURS['panel'])
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'$x$', color=COLOURS['text'], fontsize=12)
    ax1.set_ylabel(r'$y$', color=COLOURS['text'], fontsize=12)
    _style_axis(ax1)
    
    # Quarter circle
    theta = np.linspace(0, np.pi/2, 200)
    ax1.plot(np.cos(theta), np.sin(theta), color=COLOURS['text'], linewidth=2, zorder=10)
    ax1.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=COLOURS['text_dim'],
             linewidth=1, linestyle='--', alpha=0.5)
    
    # Scatter artists
    scatter_inside = ax1.scatter([], [], c=COLOURS['inside'], s=2, alpha=0.6)
    scatter_outside = ax1.scatter([], [], c=COLOURS['outside'], s=2, alpha=0.6)
    title1 = ax1.set_title('n = 0', fontweight='bold', color=COLOURS['text'], fontsize=13)
    
    # Right panel: convergence
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLOURS['panel'])
    ax2.set_xlim(1, n_points)
    ax2.set_ylim(2.8, 3.5)
    ax2.set_xscale('log')
    ax2.axhline(y=np.pi, color=COLOURS['theoretical'], linestyle='-', linewidth=2, zorder=5)
    ax2.fill_between([1, n_points], np.pi - 0.01, np.pi + 0.01, alpha=0.15, color=COLOURS['inside'])
    ax2.set_xlabel('Number of Samples (n)', color=COLOURS['text'], fontsize=12)
    ax2.set_ylabel(r'$\hat{\pi}$ Estimate', color=COLOURS['text'], fontsize=12)
    ax2.grid(True, alpha=0.3, color=COLOURS['grid'])
    _style_axis(ax2)
    
    convergence_line, = ax2.plot([], [], color=COLOURS['accent'], linewidth=1.5)
    
    # Pi estimate text
    pi_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=16,
                       color=COLOURS['text'], fontweight='bold', family='monospace',
                       verticalalignment='top')
    error_text = ax2.text(0.02, 0.85, '', transform=ax2.transAxes, fontsize=11,
                          color=COLOURS['text_dim'], verticalalignment='top')
    
    # Animation state
    state = {
        'n': 0,
        'n_inside': 0,
        'estimates': [],
        'sample_counts': []
    }
    
    def init():
        scatter_inside.set_offsets(np.empty((0, 2)))
        scatter_outside.set_offsets(np.empty((0, 2)))
        convergence_line.set_data([], [])
        return scatter_inside, scatter_outside, convergence_line, pi_text, error_text
    
    def update(frame):
        start_idx = state['n']
        end_idx = min(start_idx + batch_size, n_points)
        
        if start_idx >= n_points:
            return scatter_inside, scatter_outside, convergence_line, pi_text, error_text
        
        state['n'] = end_idx
        state['n_inside'] = np.sum(all_inside[:end_idx])
        
        # Update scatter plots
        inside_mask = all_inside[:end_idx]
        scatter_inside.set_offsets(np.c_[all_x[:end_idx][inside_mask], all_y[:end_idx][inside_mask]])
        scatter_outside.set_offsets(np.c_[all_x[:end_idx][~inside_mask], all_y[:end_idx][~inside_mask]])
        
        # Calculate pi estimate
        pi_est = 4 * state['n_inside'] / end_idx
        error = abs(pi_est - np.pi)
        
        state['estimates'].append(pi_est)
        state['sample_counts'].append(end_idx)
        
        # Update convergence line
        convergence_line.set_data(state['sample_counts'], state['estimates'])
        convergence_line.set_color(get_accuracy_colour(error))
        
        # Update text
        title1.set_text(f'n = {end_idx:,}')
        pi_text.set_text(f'pi = {pi_est:.8f}')
        pi_text.set_color(get_accuracy_colour(error))
        error_text.set_text(f'Error: {error:.2e} ({100*error/np.pi:.4f}%)')
        
        return scatter_inside, scatter_outside, convergence_line, pi_text, error_text
    
    n_frames = (n_points + batch_size - 1) // batch_size
    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                         interval=interval_ms, blit=False, repeat=False)
    
    plt.show()
    
    # Final statistics
    if state['n'] > 0:
        final_pi = 4 * state['n_inside'] / state['n']
        final_error = abs(final_pi - np.pi)
        print(f"\nSimulation complete:")
        print(f"  Final estimate: pi = {final_pi:.10f}")
        print(f"  True value:     pi = {np.pi:.10f}")
        print(f"  Error: {final_error:.2e} ({100*final_error/np.pi:.4f}%)")


def _style_axis(ax) -> None:
    """Apply consistent styling to an axis."""
    ax.tick_params(colors=COLOURS['text'], which='both')
    for spine in ax.spines.values():
        spine.set_color(COLOURS['grid'])
        spine.set_linewidth(0.8)


def main():
    """Main entry point with menu."""
    print("=" * 60)
    print("MONTE CARLO PI ESTIMATION - PUBLICATION QUALITY")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Static publication-quality figure")
    print("  2. Real-time animation")
    print("  3. Both")
    print()
    
    choice = input("Select option [1/2/3]: ").strip()
    
    if choice in ('1', '3'):
        print("\nGenerating publication figure...")
        create_publication_figure(
            n_points=25000,
            max_convergence_points=500000,
            seed=42
        )
    
    if choice in ('2', '3'):
        print("\nStarting real-time animation...")
        print("(Close the window to see final statistics)")
        run_realtime_animation(
            n_points=10000,
            batch_size=100,
            interval_ms=30,
            seed=None  # Random each time
        )


if __name__ == "__main__":
    main()
