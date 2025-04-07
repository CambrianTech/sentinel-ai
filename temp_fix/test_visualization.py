import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

# Create sample data that simulates what would be in metrics_history
def create_test_data(num_points=500):
    """Create test data to simulate a training run with many steps"""
    np.random.seed(42)
    steps = np.arange(1, num_points + 1)
    
    # Create random metrics with some patterns
    train_loss = 0.5 + 0.5 * np.exp(-steps/100) + np.random.normal(0, 0.05, num_points)
    eval_loss = 0.4 + 0.6 * np.exp(-steps/120) + np.random.normal(0, 0.07, num_points)
    
    # Create pruning events (occasional spikes)
    pruned_heads = np.zeros(num_points)
    for i in range(10, num_points, 50):
        if i < num_points:
            pruned_heads[i] = np.random.randint(1, 5)
    
    # Rarely revive heads
    revived_heads = np.zeros(num_points)
    for i in range(100, num_points, 150):
        if i < num_points:
            revived_heads[i] = np.random.randint(0, 2)
    
    # Increasing sparsity that plateaus
    sparsity = 0.1 + 0.4 * (1 - np.exp(-steps/250)) + np.random.normal(0, 0.01, num_points)
    
    # Perplexity - generally goes down but with some noise
    perplexity = 20 + 30 * np.exp(-steps/200) + np.random.normal(0, 1.5, num_points)
    
    # Create epochs - change every ~100 steps
    epoch = np.ones(num_points, dtype=int)
    for i in range(1, num_points):
        if i % 100 == 0:
            epoch[i:] = epoch[i-1] + 1
    
    # Compile into metrics_history format
    metrics_history = {
        "step": steps.tolist(),
        "train_loss": train_loss.tolist(),
        "eval_loss": eval_loss.tolist(),
        "pruned_heads": pruned_heads.tolist(),
        "revived_heads": revived_heads.tolist(), 
        "sparsity": sparsity.tolist(),
        "epoch": epoch.tolist(),
        "perplexity": perplexity.tolist()
    }
    
    return metrics_history

# Test our visualization code
def test_visualization():
    # Create test data
    metrics_history = create_test_data(500)  # Generate 500 data points
    
    # Save the original version of the plot (without our fix)
    create_original_plot(metrics_history)
    
    # Save the fixed version of the plot
    create_fixed_plot(metrics_history)
    
    print("Generated both plots. Check 'original_plot.png' and 'fixed_plot.png'")

def create_original_plot(metrics_history):
    """Create a plot using the original code that might create huge images"""
    print("Creating original plot (might be very large)...")
    
    try:
        # Create plot with default settings - might be very large
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # Plot losses
        ax1.plot(metrics_history["step"], metrics_history["train_loss"], label="Train Loss")
        ax1.plot(metrics_history["step"], metrics_history["eval_loss"], label="Eval Loss")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Evaluation Loss")
        ax1.legend()
        ax1.grid(True)

        # Mark epoch boundaries if available
        if "epoch" in metrics_history and len(metrics_history["epoch"]) > 1:
            for i in range(1, len(metrics_history["epoch"])):
                if metrics_history["epoch"][i] != metrics_history["epoch"][i-1]:
                    # This is an epoch boundary
                    for ax in [ax1, ax2, ax3]:
                        ax.axvline(x=metrics_history["step"][i], color="k", linestyle="--", alpha=0.3)
                        ax.text(metrics_history["step"][i], ax.get_ylim()[1]*0.9, 
                                f"Epoch {metrics_history['epoch'][i]}", rotation=90, alpha=0.7)

        # Plot pruning metrics
        ax2.bar(metrics_history["step"], metrics_history["pruned_heads"], alpha=0.5, label="Pruned Heads", color="blue")
        ax2.bar(metrics_history["step"], metrics_history["revived_heads"], alpha=0.5, label="Revived Heads", color="green")
        ax2.set_ylabel("Count")
        ax2.set_title("Head Pruning and Revival")
        ax2.legend(loc="upper left")
        ax2.grid(True)

        # Plot sparsity and perplexity
        ax3.plot(metrics_history["step"], metrics_history["sparsity"], "r-", label="Sparsity")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Sparsity")
        ax3.grid(True)

        # Add perplexity line on secondary axis if available
        if "perplexity" in metrics_history and metrics_history["perplexity"]:
            ax4 = ax3.twinx()
            ax4.plot(metrics_history["step"], metrics_history["perplexity"], "g-", label="Perplexity")
            ax4.set_ylabel("Perplexity")
            ax4.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig('original_plot.png')
        plt.close()
        print(f"Original plot created with size ({len(metrics_history['step'])} points)")
    except Exception as e:
        print(f"Error creating original plot: {e}")

def create_fixed_plot(metrics_history):
    """Create a plot using our fixed code with downsampling"""
    print("Creating fixed plot with downsampling...")
    
    try:
        # Visualize training metrics with epochs - FIXED VERSION
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), dpi=100, sharex=True)

        # Set maximum display limit to prevent excessively large plots
        max_display_points = 100
        display_steps = metrics_history["step"]
        if len(display_steps) > max_display_points:
            # Downsample by selecting evenly spaced points
            indices = np.linspace(0, len(display_steps) - 1, max_display_points).astype(int)
            display_steps = [metrics_history["step"][i] for i in indices]
            display_train_loss = [metrics_history["train_loss"][i] for i in indices]
            display_eval_loss = [metrics_history["eval_loss"][i] for i in indices]
            display_pruned_heads = [metrics_history["pruned_heads"][i] for i in indices]
            display_revived_heads = [metrics_history["revived_heads"][i] for i in indices]
            display_sparsity = [metrics_history["sparsity"][i] for i in indices]
            display_epoch = [metrics_history["epoch"][i] for i in indices]
            display_perplexity = [metrics_history["perplexity"][i] for i in indices] if "perplexity" in metrics_history and metrics_history["perplexity"] else []
        else:
            display_train_loss = metrics_history["train_loss"]
            display_eval_loss = metrics_history["eval_loss"]
            display_pruned_heads = metrics_history["pruned_heads"]
            display_revived_heads = metrics_history["revived_heads"]
            display_sparsity = metrics_history["sparsity"]
            display_epoch = metrics_history["epoch"]
            display_perplexity = metrics_history["perplexity"] if "perplexity" in metrics_history else []

        # Plot losses
        ax1.plot(display_steps, display_train_loss, label="Train Loss")
        ax1.plot(display_steps, display_eval_loss, label="Eval Loss")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Evaluation Loss")
        ax1.legend()
        ax1.grid(True)

        # Mark epoch boundaries if available
        if "epoch" in metrics_history and len(display_epoch) > 1:
            for i in range(1, len(display_epoch)):
                if display_epoch[i] != display_epoch[i-1]:
                    # This is an epoch boundary
                    for ax in [ax1, ax2, ax3]:
                        ax.axvline(x=display_steps[i], color="k", linestyle="--", alpha=0.3)
                        ax.text(display_steps[i], ax.get_ylim()[1]*0.9, 
                                f"Epoch {display_epoch[i]}", rotation=90, alpha=0.7)

        # Plot pruning metrics
        ax2.bar(display_steps, display_pruned_heads, alpha=0.5, label="Pruned Heads", color="blue")
        ax2.bar(display_steps, display_revived_heads, alpha=0.5, label="Revived Heads", color="green")
        ax2.set_ylabel("Count")
        ax2.set_title("Head Pruning and Revival")
        ax2.legend(loc="upper left")
        ax2.grid(True)

        # Plot sparsity and perplexity
        ax3.plot(display_steps, display_sparsity, "r-", label="Sparsity")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Sparsity")
        ax3.grid(True)

        # Add perplexity line on secondary axis if available
        if "perplexity" in metrics_history and metrics_history["perplexity"]:
            ax4 = ax3.twinx()
            ax4.plot(display_steps, display_perplexity, "g-", label="Perplexity")
            ax4.set_ylabel("Perplexity")
            ax4.legend(loc="upper right")

        # Ensure figure has reasonable dimensions
        plt.gcf().set_dpi(100)
        plt.tight_layout()
        plt.savefig('fixed_plot.png')
        plt.close()
        print(f"Fixed plot created with {len(display_steps)} points (downsampled from {len(metrics_history['step'])})")
    except Exception as e:
        print(f"Error creating fixed plot: {e}")

if __name__ == "__main__":
    test_visualization()