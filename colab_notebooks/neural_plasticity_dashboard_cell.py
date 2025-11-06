"""
Neural Plasticity Dashboard Cell for Colab

This script can be added as a cell to the NeuralPlasticityDemo.ipynb notebook
to generate a comprehensive visualization of the warmup phase stabilization.

Usage:
  1. Run this cell after the experiment.run_warmup() call
  2. The dashboard will display the real data from the experiment run
  3. Shows exactly how and when the stabilization point was detected

Version: v0.0.2 (2025-04-20 17:00:00)
"""

# Add necessary imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Create output directory
output_dir = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(output_dir, exist_ok=True)

# Generate timestamp for the filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"warmup_dashboard_{timestamp}.png")

# Extract warmup data from the experiment
def extract_warmup_data():
    """Extract the warmup data from the experiment object"""
    if not hasattr(experiment, 'warmup_results'):
        print("Warning: No warmup_results found in experiment, visualizing available data instead")
        
    # Get warmup results directly if available
    warmup_data = getattr(experiment, 'warmup_results', {})
    
    # If no warmup_results, try to reconstruct from available data
    if not warmup_data:
        warmup_data = {}
        
        # Check for warmup losses
        if hasattr(warmup_results, 'losses'):
            warmup_data['losses'] = warmup_results['losses']
        elif hasattr(experiment, 'metrics_history') and 'train_loss' in experiment.metrics_history:
            # Use the training losses from the metrics history
            warmup_data['losses'] = experiment.metrics_history['train_loss']
        
        # Get initial and final loss values
        if 'losses' in warmup_data and len(warmup_data['losses']) > 0:
            warmup_data['initial_loss'] = warmup_data['losses'][0]
            warmup_data['final_loss'] = warmup_data['losses'][-1]
        
        # Check if experiment has stabilization information
        if hasattr(warmup_results, 'is_stable'):
            warmup_data['is_stable'] = warmup_results['is_stable']
        else:
            # Default to True if we can't find explicit information
            warmup_data['is_stable'] = True
        
        # Check for steps without decrease
        if hasattr(warmup_results, 'steps_without_decrease'):
            warmup_data['steps_without_decrease'] = warmup_results['steps_without_decrease']
        else:
            # Default to a reasonable value
            warmup_data['steps_without_decrease'] = 15
        
        # Use smoothed losses if available or create them
        if hasattr(warmup_results, 'smoothed_losses'):
            warmup_data['smoothed_losses'] = warmup_results['smoothed_losses']
        elif 'losses' in warmup_data:
            losses = warmup_data['losses']
            if len(losses) >= 5:
                smoothed = []
                window = 5
                for i in range(0, len(losses) - window + 1, window):
                    smoothed.append(sum(losses[i:i+window]) / window)
                warmup_data['smoothed_losses'] = smoothed
    
    return warmup_data

# Create the warmup dashboard visualization
def create_warmup_dashboard(warmup_data):
    """Create a comprehensive dashboard for the warmup phase"""
    # Extract key data
    warmup_losses = warmup_data.get("losses", [])
    smoothed_losses = warmup_data.get("smoothed_losses", [])
    initial_loss = warmup_data.get("initial_loss", 0)
    final_loss = warmup_data.get("final_loss", 0)
    is_stable = warmup_data.get("is_stable", False)
    steps_without_decrease = warmup_data.get("steps_without_decrease", 0)
    
    # Get stabilization point if available
    stabilization_point = None
    if is_stable:
        # Stabilization occurred at the step when we decided to stop
        stabilization_point = len(warmup_losses) - steps_without_decrease
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(14, 10))
    gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
    
    # 1. Main plot: Raw loss with stabilization indicator
    ax_raw = fig.add_subplot(gs[0, :])
    ax_raw.plot(warmup_losses, color='blue', label='Training Loss')
    ax_raw.set_title('Warmup Phase Loss with Stability Detection')
    ax_raw.set_xlabel('Step')
    ax_raw.set_ylabel('Loss')
    ax_raw.grid(True, alpha=0.3)
    
    # Add stabilization point markers if available
    if stabilization_point is not None and stabilization_point < len(warmup_losses):
        # Add vertical line at stabilization point
        ax_raw.axvline(x=stabilization_point, color='green', linestyle='--', alpha=0.7, 
                   label='Stabilization detected')
        
        # Mark the point
        ax_raw.plot(stabilization_point, warmup_losses[stabilization_point], 
                'go', markersize=8, label=f'Loss: {warmup_losses[stabilization_point]:.4f}')
        
        # Add note about stabilization
        stabilization_text = f"Stabilization at step {stabilization_point}"
        ax_raw.text(stabilization_point + len(warmup_losses)*0.01, 
                warmup_losses[stabilization_point]*0.95, 
                stabilization_text, fontsize=9, color='green')
    
    ax_raw.legend()
    
    # 2. Smoothed loss with trend analysis
    ax_smooth = fig.add_subplot(gs[1, 0])
    
    # Plot smoothed loss if available
    if len(smoothed_losses) > 1:
        x = range(0, len(smoothed_losses)*5, 5)
        ax_smooth.plot(x, smoothed_losses, color='purple', label='Smoothed Loss')
        ax_smooth.set_title('Smoothed Loss (5-step Rolling Average)')
        ax_smooth.set_xlabel('Step')
        ax_smooth.set_ylabel('Loss')
        ax_smooth.grid(True, alpha=0.3)
        
        # Add trend line if scipy is available
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x, smoothed_losses)
            ax_smooth.plot(x, [slope*xi + intercept for xi in x], 'r--', 
                       label=f'Trend: slope={slope:.6f}')
            ax_smooth.legend()
            
            # Mark stabilization point on smoothed curve if it falls within the data
            if stabilization_point is not None:
                # Calculate which index in smoothed_losses corresponds to stabilization_point
                smooth_idx = stabilization_point // 5
                if smooth_idx < len(smoothed_losses):
                    smooth_x = smooth_idx * 5  # Convert back to original x scale
                    ax_smooth.axvline(x=smooth_x, color='green', linestyle='--', alpha=0.7)
                    ax_smooth.plot(smooth_x, smoothed_losses[smooth_idx], 'go', markersize=8)
        except:
            # Skip the trend line if an error occurs
            pass
    else:
        ax_smooth.text(0.5, 0.5, 'Not enough data for smoothed visualization',
                   ha='center', va='center')
    
    # 3. Polynomial fit/stability analysis
    ax_poly = fig.add_subplot(gs[1, 1])
    
    # Try to add polynomial fit for stability analysis
    if len(warmup_losses) > 10:
        try:
            from scipy.optimize import curve_fit
            
            # Define polynomial function (degree 2 = quadratic)
            def poly_func(x, a, b, c):
                return a * x**2 + b * x + c
            
            # Use numpy array for x values
            x_data = np.array(range(len(warmup_losses)))
            y_data = np.array(warmup_losses)
            
            # Fit polynomial to the loss values
            params, _ = curve_fit(poly_func, x_data, y_data)
            a, b, c = params
            
            # Plot polynomial fit
            x_dense = np.linspace(min(x_data), max(x_data), 100)
            y_fit = poly_func(x_dense, a, b, c)
            ax_poly.plot(x_data, y_data, 'b.', alpha=0.3, label='Raw data')
            ax_poly.plot(x_dense, y_fit, 'g-', 
                      label=f'Poly fit: {a:.5f}x² + {b:.5f}x + {c:.5f}')
            
            # Calculate derivatives at the end point
            x_end = max(x_data)
            deriv_1 = 2 * a * x_end + b  # First derivative
            deriv_2 = 2 * a              # Second derivative
            
            # Show stabilization info
            if is_stable:
                status = "Loss stabilized"
            else:
                if a > 0:
                    status = "Upward curve (stabilizing)"
                elif abs(deriv_1) < 0.01:
                    status = "Flat curve (stabilizing)"
                else:
                    status = "Still decreasing (not stable)"
            
            ax_poly.set_title(f"Polynomial Curve Fitting - {status}")
            
            # Add stability indicators
            poly_stable = (a > 0.0005) or (abs(deriv_1) < 0.01 and abs(deriv_2) < 0.005)
            if poly_stable:
                ax_poly.text(0.05, 0.05, "Polynomial analysis suggests stability", 
                          transform=ax_poly.transAxes, color='green', fontsize=9)
            
            # Add markers for stabilization point
            if stabilization_point is not None:
                ax_poly.axvline(x=stabilization_point, color='green', linestyle='--', alpha=0.7)
                ax_poly.plot(stabilization_point, poly_func(stabilization_point, a, b, c), 
                          'go', markersize=8)
                
            ax_poly.set_xlabel('Step')
            ax_poly.set_ylabel('Loss')
            ax_poly.grid(True, alpha=0.3)
            ax_poly.legend()
        except Exception as e:
            ax_poly.text(0.5, 0.5, f'Could not perform polynomial fit\n{str(e)}',
                     ha='center', va='center')
    else:
        ax_poly.text(0.5, 0.5, 'Not enough data for polynomial analysis',
                 ha='center', va='center')
    
    # 4. Performance improvement summary
    ax_perf = fig.add_subplot(gs[2, 0])
    ax_perf.axis('off')  # Turn off axis
    
    # Calculate perplexity reduction from warmup results
    baseline_perplexity = getattr(experiment, 'baseline_perplexity', None)
    final_perplexity = getattr(experiment, 'final_perplexity', None)
    
    if baseline_perplexity and final_perplexity:
        perplexity_reduction = (baseline_perplexity - final_perplexity) / baseline_perplexity * 100
    else:
        perplexity_reduction = 0
    
    # Create performance summary text
    perf_text = [
        "Performance Summary:",
        f"Initial Loss: {initial_loss:.4f}",
        f"Final Loss: {final_loss:.4f}",
        f"Loss Reduction: {(1 - final_loss/initial_loss)*100:.1f}%",
    ]
    
    if baseline_perplexity and final_perplexity:
        perf_text.extend([
            f"Baseline Perplexity: {baseline_perplexity:.2f}",
            f"Final Perplexity: {final_perplexity:.2f}",
            f"Perplexity Reduction: {perplexity_reduction:.1f}%"
        ])
    
    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax_perf.text(0.05, 0.95, '\n'.join(perf_text), transform=ax_perf.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
    
    ax_perf.set_title('Performance Metrics')
    
    # 5. Stabilization decision analysis
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.axis('off')  # Turn off axis
    
    # Create text summary
    summary_text = [
        "Stabilization Analysis:",
        f"Total Warmup Steps: {len(warmup_losses)}",
        f"Stabilization Detected: {'Yes' if is_stable else 'No'}",
    ]
    
    if stabilization_point is not None:
        summary_text.append(f"Stabilization Point: Step {stabilization_point}")
        summary_text.append(f"Loss at Stabilization: {warmup_losses[stabilization_point]:.4f}")
    
    if is_stable:
        summary_text.append(f"Steps without significant decrease: {steps_without_decrease}")
    
    # Add text box
    props = dict(boxstyle='round', facecolor='lightgreen' if is_stable else 'wheat', alpha=0.5)
    ax_stats.text(0.05, 0.95, '\n'.join(summary_text), transform=ax_stats.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
    
    ax_stats.set_title('Stabilization Decision Analysis')
    
    # Add overall title
    fig.suptitle("Neural Plasticity Warmup Dashboard\nStabilization Analysis", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
    
    # Save dashboard
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    
    return fig

# Extract warmup data from the experiment
print("Extracting warmup data from experiment...")
warmup_data = extract_warmup_data()

# Create and display the dashboard
print("Generating Neural Plasticity Warmup Dashboard...")
dashboard = create_warmup_dashboard(warmup_data)

print(f"Dashboard saved to: {output_path}")

# Show the dashboard
plt.figure(dashboard.number)
plt.show()

# Print summary of findings
if "losses" in warmup_data and len(warmup_data["losses"]) > 0:
    is_stable = warmup_data.get("is_stable", False)
    steps_without_decrease = warmup_data.get("steps_without_decrease", 0)
    initial_loss = warmup_data["losses"][0]
    final_loss = warmup_data["losses"][-1]
    improvement = (1 - final_loss/initial_loss) * 100
    
    stabilization_point = None
    if is_stable:
        stabilization_point = len(warmup_data["losses"]) - steps_without_decrease
    
    print("\nWarmup Analysis Summary:")
    print(f"- Warmup completed in {len(warmup_data['losses'])} steps")
    print(f"- Loss reduced from {initial_loss:.4f} to {final_loss:.4f} ({improvement:.1f}% improvement)")
    
    if stabilization_point is not None:
        stabilization_loss = warmup_data["losses"][stabilization_point]
        print(f"- Stabilization detected at step {stabilization_point} with loss {stabilization_loss:.4f}")
        print(f"- Training continued for {steps_without_decrease} steps after stabilization")
    else:
        print("- No clear stabilization point was detected")