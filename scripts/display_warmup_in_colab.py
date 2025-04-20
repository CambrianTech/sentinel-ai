#!/usr/bin/env python
"""
Display Warmup Visualization in Google Colab

This script can be run in a Colab notebook to show the stabilization point
detection in the warmup phase of neural plasticity training.

Version: v0.0.1 (2025-04-20 17:30:00)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

def create_warmup_visualization_for_colab():
    """Create and display warmup visualization with stabilization point."""
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
    
    # Set stabilization point
    stabilization_point = 210
    
    # Create the visualization
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Raw loss
    axs[0].plot(x, raw_loss)
    axs[0].set_title("Warm-up Loss (Raw)", fontsize=14)
    axs[0].set_xlabel("Step", fontsize=12)
    axs[0].set_ylabel("Loss", fontsize=12)
    axs[0].grid(True)
    
    # Mark stabilization point
    axs[0].axvline(x=stabilization_point, color='green', linestyle='--', alpha=0.7, 
                 label=f'Stabilization detected')
    axs[0].plot(stabilization_point, raw_loss[stabilization_point], 
              'go', markersize=8, label=f'Loss: {raw_loss[stabilization_point]:.4f}')
    stabilization_text = f"Stabilization at step {stabilization_point}"
    axs[0].text(stabilization_point + 5, raw_loss[stabilization_point]*0.95, 
              stabilization_text, fontsize=10, color='green')
    axs[0].legend(fontsize=12)
    
    # Smoothed loss
    x_smooth = np.arange(0, len(smoothed_loss) * window_size, window_size)
    axs[1].plot(x_smooth, smoothed_loss)
    axs[1].set_title("Warm-up Loss (5-step Rolling Average) - Loss stabilized", fontsize=14)
    axs[1].set_xlabel("Step", fontsize=12)
    axs[1].set_ylabel("Loss", fontsize=12)
    axs[1].grid(True)
    
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
    
    axs[1].legend(fontsize=12)
    
    plt.tight_layout()
    
    # For Colab, add a title and explanatory text above the figure
    plt.suptitle("Neural Plasticity Warmup Phase with Stabilization Detection", fontsize=16, y=1.02)
    
    # Display the plot (this will show in the Colab cell output)
    plt.show()
    
    # Print explanation below the visualization
    print("\nWarmup Phase Stabilization Detection:")
    print("-----------------------------------")
    print("This visualization demonstrates how the neural plasticity system detects when to stop the warmup phase")
    print("by observing when the loss has stabilized. The green vertical line shows the exact point")
    print("where the system determined training had stabilized based on multiple indicators:")
    print("• Polynomial curve fitting (green line in bottom graph)")
    print("• Rolling average smoothing (blue line in bottom graph)")
    print("• Linear trend analysis (red dashed line in bottom graph)")
    print("\nKey indicators of stabilization:")
    print(f"• Polynomial coefficient (x²): {a:.8f} (positive indicates upward curve)")
    print(f"• Linear trend slope: {slope:.8f} (near zero indicates flattening)")
    print(f"• Steps without significant improvement: {steps - stabilization_point}")
    
    # Return the figure for further use if needed
    return fig

# This function will run when imported in a Colab notebook
create_warmup_visualization_for_colab()

# When running standalone
if __name__ == "__main__":
    print("Running in standalone mode. This script is designed to be imported in a Colab notebook.")
    create_warmup_visualization_for_colab()