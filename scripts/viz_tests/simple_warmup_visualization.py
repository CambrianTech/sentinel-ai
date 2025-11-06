#!/usr/bin/env python
"""
Simple Warmup Visualization

This script generates a visualization showing the warmup phase
with the stabilization point clearly marked.

Version: v0.0.1 (2025-04-20 15:55:00)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_warmup_visualization(
    stabilization_point=None, 
    output_path="warmup_visualization.png"
):
    """
    Create visualization for warmup training with stabilization point.
    
    Args:
        stabilization_point: Step at which stability was detected
        output_path: Path to save the visualization
    """
    # Generate sample data
    steps = 250
    x = np.arange(steps)
    
    # Create loss curve that stabilizes
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 0.05, steps)
    
    # Exponential decay plus noise plus slight upward curve at the end
    raw_loss = 7.5 * np.exp(-0.01 * x) + 4.5 + noise
    
    # Add a slight upward curve starting at step 200
    raw_loss[200:] += 0.0001 * (x[200:] - 200)**2
    
    # Calculate smoothed loss (5-step windows)
    window_size = 5
    smoothed_loss = []
    for i in range(0, steps, window_size):
        if i+window_size <= steps:
            smoothed_loss.append(np.mean(raw_loss[i:i+window_size]))
    
    # Set default stabilization point if not provided
    if stabilization_point is None:
        stabilization_point = 210
    
    # Create the visualization
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    
    # Raw loss
    axs[0].plot(x, raw_loss)
    axs[0].set_title("Warm-up Loss (Raw)")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True)
    
    # Mark stabilization point
    axs[0].axvline(x=stabilization_point, color='green', linestyle='--', alpha=0.7, 
                 label=f'Stabilization detected')
    axs[0].plot(stabilization_point, raw_loss[stabilization_point], 
              'go', markersize=8, label=f'Loss: {raw_loss[stabilization_point]:.4f}')
    stabilization_text = f"Stabilization at step {stabilization_point}"
    axs[0].text(stabilization_point + 5, raw_loss[stabilization_point]*0.95, 
              stabilization_text, fontsize=9, color='green')
    axs[0].legend()
    
    # Smoothed loss
    x_smooth = np.arange(0, len(smoothed_loss) * window_size, window_size)
    axs[1].plot(x_smooth, smoothed_loss)
    axs[1].set_title("Warm-up Loss (5-step Rolling Average) - Loss stabilized")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Loss")
    axs[1].grid(True)
    
    # Add trend line and polynomial fit if scipy is available
    try:
        from scipy.stats import linregress
        from scipy.optimize import curve_fit
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x_smooth, smoothed_loss)
        axs[1].plot(x_smooth, [slope*xi + intercept for xi in x_smooth], 'r--', 
                 label=f'Trend: slope={slope:.6f}, R²={r_value**2:.2f}')
        
        # Polynomial fit
        def poly_func(x, a, b, c):
            return a * x**2 + b * x + c
        
        params, _ = curve_fit(poly_func, x_smooth, smoothed_loss)
        a, b, c = params
        
        # Plot polynomial fit
        x_dense = np.linspace(min(x_smooth), max(x_smooth), 100)
        y_fit = poly_func(x_dense, a, b, c)
        axs[1].plot(x_dense, y_fit, 'g-', 
                  label=f'Poly fit: {a:.5f}x² + {b:.5f}x + {c:.5f}')
        
        # Mark stabilization on smoothed curve
        smooth_idx = stabilization_point // window_size
        if smooth_idx < len(smoothed_loss):
            smooth_x = smooth_idx * window_size
            axs[1].axvline(x=smooth_x, color='green', linestyle='--', alpha=0.7)
            axs[1].plot(smooth_x, smoothed_loss[smooth_idx], 'go', markersize=8)
        
        axs[1].legend()
    except ImportError:
        print("scipy not available, skipping trend line and polynomial fit")
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    return fig

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate warmup visualization with stabilization point")
    parser.add_argument("--stabilization_point", type=int, default=210,
                      help="Step at which stabilization occurs")
    parser.add_argument("--output", type=str, default="warmup_visualization.png",
                      help="Path to save the visualization")
    parser.add_argument("--show", action="store_true", 
                      help="Show the visualization in a window")
    args = parser.parse_args()
    
    print(f"Generating warmup visualization with stabilization at step {args.stabilization_point}...")
    fig = create_warmup_visualization(args.stabilization_point, args.output)
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()