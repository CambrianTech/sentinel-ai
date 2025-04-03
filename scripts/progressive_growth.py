#!/usr/bin/env python
"""
Progressive Growth Implementation

This script demonstrates the Sentinel-AI framework's ability to progressively grow
transformer heads from an initially heavily pruned state. Unlike conventional approaches
that start with full models and prune, this shows how models can:

1. Start in a highly efficient, heavily pruned state
2. Strategically regrow attention heads based on gradient and importance signals
3. Evolve into more powerful models while maintaining efficiency
4. Target growth to the most valuable computational pathways

This capability is critical for creating models that can grow into more powerful
systems based on task demands, rather than needing to start with overparameterized
architectures.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from data_modules.dataset_loader import load_dataset
from utils.model_wrapper import wrap_model_for_generation
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.generation_wrapper import generate_text
from controller.controller_manager import ControllerManager
from controller.metrics.head_metrics import collect_head_metrics
from utils.head_metrics import compute_head_importance


def parse_args():
    parser = argparse.ArgumentParser(description="Progressive Growth with Sentinel-AI")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Model name or path (e.g., distilgpt2, gpt2)")
    
    # Initial pruning configuration
    parser.add_argument("--initial_pruning", type=float, default=0.9,
                        help="Initial pruning level to start from (0.0-0.99)")
    parser.add_argument("--initial_strategy", type=str, default="gradient",
                        choices=["random", "entropy", "gradient", "uniform"],
                        help="Initial pruning strategy (determines which heads remain)")
    
    # Growth configuration
    parser.add_argument("--growth_rate", type=float, default=0.1,
                        help="Growth rate per epoch (fraction of total heads)")
    parser.add_argument("--growth_strategy", type=str, default="importance",
                        choices=["importance", "gradient", "entropy", "random"],
                        help="Strategy for selecting heads to regrow")
    parser.add_argument("--target_pruning", type=float, default=0.3,
                        help="Target pruning level to reach (final state)")
    
    # Task and dataset
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                        choices=["tiny_shakespeare", "wikitext", "tiny_stories"],
                        help="Dataset to use for training")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate every N steps")
    
    # U-Net skip connections
    parser.add_argument("--enable_unet", action="store_true",
                        help="Enable U-Net style skip connections")
    parser.add_argument("--connection_scale", type=float, default=0.05,
                        help="Scaling factor for U-Net skip connections")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="growth_results",
                        help="Directory to save results")
    parser.add_argument("--drive_path", type=str, default="",
                        help="Google Drive path for saving results (for Colab)")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def apply_initial_pruning(model, strategy, pruning_level, device):
    """
    Apply initial heavy pruning to the model before starting progressive growth.
    
    Args:
        model: The adaptive transformer model
        strategy: Pruning strategy to use ("random", "entropy", "gradient", "uniform")
        pruning_level: Initial pruning level (0.0-0.99)
        device: Device to use for calculations
        
    Returns:
        Pruned model
    """
    print(f"Applying initial {strategy} pruning at {pruning_level:.1%} level")
    
    # Get model dimensions
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    total_heads = num_layers * num_heads
    heads_to_keep = int(total_heads * (1 - pruning_level))
    
    if heads_to_keep < 1:
        print("Warning: Pruning level too high, keeping at least 1 head per layer")
        heads_to_keep = num_layers  # Ensure at least 1 head per layer

    # Create dummy input for collecting metrics if needed
    batch_size = 2
    seq_len = 32
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    dummy_batch = {"input_ids": dummy_input, 
                  "attention_mask": torch.ones_like(dummy_input)}
    
    # Set all gates to near-zero initially
    with torch.no_grad():
        for l in range(num_layers):
            for h in range(num_heads):
                model.blocks[l]["attn"].gate[h] = torch.tensor(0.001, device=device)
    
    # Apply pruning based on strategy
    if strategy == "random":
        # Get a flattened list of (layer, head) tuples
        all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
        
        # Randomly select heads to keep active
        kept_head_indices = np.random.choice(len(all_heads), heads_to_keep, replace=False)
        
        # Set gates to 1.0 for kept heads
        with torch.no_grad():
            for idx in kept_head_indices:
                layer_idx, head_idx = all_heads[idx]
                model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(1.0, device=device)
    
    elif strategy == "uniform":
        # Distribute active heads uniformly across layers
        heads_per_layer = max(1, heads_to_keep // num_layers)
        remaining_heads = heads_to_keep - (heads_per_layer * num_layers)
        
        with torch.no_grad():
            for layer_idx in range(num_layers):
                # Determine how many heads to keep in this layer
                layer_heads = heads_per_layer
                if layer_idx < remaining_heads:
                    layer_heads += 1
                
                # Randomly select heads to keep in this layer
                head_indices = np.random.choice(num_heads, layer_heads, replace=False)
                
                # Set gates to 1.0 for kept heads
                for head_idx in head_indices:
                    model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(1.0, device=device)
    
    elif strategy in ["entropy", "gradient"]:
        # Collect metrics
        metrics = collect_head_metrics(model, batch=dummy_batch)
        
        if strategy == "entropy" and "entropy" in metrics:
            head_scores = metrics["entropy"]
            # Lower entropy = more focused attention = more important to keep
            descending = False
        elif strategy == "gradient" and "grad_norm" in metrics:
            head_scores = metrics["grad_norm"]
            # Higher gradient norm = more important head = more important to keep
            descending = True
        else:
            print(f"Warning: {strategy} metrics not available, using random pruning")
            return apply_initial_pruning(model, "random", pruning_level, device)
        
        # Reshape and flatten scores
        if not isinstance(head_scores, torch.Tensor):
            head_scores = torch.tensor(head_scores, device=device)
            
        if len(head_scores.shape) < 2:
            head_scores = head_scores.reshape(num_layers, num_heads)
            
        flat_scores = head_scores.view(-1)
        
        # Sort scores
        _, indices = torch.sort(flat_scores, descending=descending)
        indices_to_keep = indices[:heads_to_keep]
        
        # Apply pruning - keep only selected heads
        with torch.no_grad():
            # First set all gates to 0.001 (pruned)
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=device)
            
            # Then activate only the selected heads
            for idx in indices_to_keep:
                layer_idx = idx.item() // num_heads
                head_idx = idx.item() % num_heads
                model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(1.0, device=device)
    
    # Count active heads for verification
    active_count = 0
    with torch.no_grad():
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                if model.blocks[layer_idx]["attn"].gate[head_idx].item() > 0.5:
                    active_count += 1
    
    print(f"Kept {active_count} of {total_heads} heads active ({active_count/total_heads:.1%})")
    return model


def get_head_growth_order(model, strategy, dataloader, device):
    """
    Determine the order in which to grow attention heads based on the chosen strategy.
    
    Args:
        model: The adaptive transformer model
        strategy: Growth strategy to use ("importance", "gradient", "entropy", "random")
        dataloader: Dataloader with training data for importance metrics
        device: Device to use for calculations
        
    Returns:
        List of (layer_idx, head_idx) tuples ordered by growth priority
    """
    # Get model dimensions
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    
    # Get currently inactive heads
    inactive_heads = []
    with torch.no_grad():
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                if model.blocks[layer_idx]["attn"].gate[head_idx].item() < 0.5:
                    inactive_heads.append((layer_idx, head_idx))
    
    if strategy == "random":
        # Shuffle the inactive heads randomly
        np.random.shuffle(inactive_heads)
        return inactive_heads
    
    # For other strategies, we need metrics for ranking
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    
    if strategy == "importance":
        # Compute head importance for regrowth
        print("Computing head importance for regrowth...")
        
        # Temporarily activate all heads for importance calculation
        head_gates_backup = {}
        with torch.no_grad():
            for layer_idx, head_idx in inactive_heads:
                # Store original gate value
                head_gates_backup[(layer_idx, head_idx)] = model.blocks[layer_idx]["attn"].gate[head_idx].item()
                # Temporarily set gate to 1.0
                model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(1.0, device=device)
        
        # Compute importance scores for all heads
        importance_scores = compute_head_importance(model, batch)
        
        # Restore original gate values
        with torch.no_grad():
            for (layer_idx, head_idx), gate_value in head_gates_backup.items():
                model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(gate_value, device=device)
        
        # Create a list of (importance, layer_idx, head_idx) for inactive heads
        head_importance = []
        for layer_idx, head_idx in inactive_heads:
            imp = importance_scores[layer_idx][head_idx].item() if isinstance(importance_scores, torch.Tensor) else importance_scores[layer_idx, head_idx]
            head_importance.append((imp, layer_idx, head_idx))
        
        # Sort by importance (higher first)
        head_importance.sort(reverse=True)
        
        # Return only the (layer_idx, head_idx) tuples in order of importance
        return [(layer_idx, head_idx) for _, layer_idx, head_idx in head_importance]
    
    elif strategy in ["gradient", "entropy"]:
        # Collect metrics
        metrics = collect_head_metrics(model, batch=batch)
        
        if strategy == "entropy" and "entropy" in metrics:
            head_scores = metrics["entropy"]
            # Lower entropy = more focused attention = higher priority for growth
            reverse = False
        elif strategy == "gradient" and "grad_norm" in metrics:
            head_scores = metrics["grad_norm"]
            # Higher gradient norm = more important head = higher priority for growth
            reverse = True
        else:
            print(f"Warning: {strategy} metrics not available, using random growth")
            return inactive_heads
        
        # Create a list of (score, layer_idx, head_idx) for inactive heads
        head_scores_list = []
        for layer_idx, head_idx in inactive_heads:
            score = head_scores[layer_idx][head_idx].item() if isinstance(head_scores, torch.Tensor) else head_scores[layer_idx, head_idx]
            head_scores_list.append((score, layer_idx, head_idx))
        
        # Sort by score
        head_scores_list.sort(reverse=reverse)
        
        # Return only the (layer_idx, head_idx) tuples in order of score
        return [(layer_idx, head_idx) for _, layer_idx, head_idx in head_scores_list]
    
    # Default to random order if strategy not recognized
    return inactive_heads


def grow_attention_heads(model, num_heads_to_grow, growth_order, device):
    """
    Grow attention heads according to the specified order.
    
    Args:
        model: The adaptive transformer model
        num_heads_to_grow: Number of heads to activate in this growth step
        growth_order: List of (layer_idx, head_idx) tuples in growth priority order
        device: Device to use for calculations
        
    Returns:
        Number of heads actually grown
    """
    if not growth_order:
        print("No more heads to grow.")
        return 0
    
    print(f"Growing {num_heads_to_grow} attention heads...")
    heads_to_grow = growth_order[:num_heads_to_grow]
    
    # Activate the selected heads
    with torch.no_grad():
        for layer_idx, head_idx in heads_to_grow:
            # Activate the head by setting its gate to 1.0
            model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(1.0, device=device)
            print(f"  Activated head {head_idx} in layer {layer_idx}")
    
    return len(heads_to_grow)


def count_active_heads(model):
    """
    Count the number of active attention heads in the model.
    
    Args:
        model: The adaptive transformer model
        
    Returns:
        Tuple of (active_head_count, total_head_count)
    """
    active_count = 0
    total_count = 0
    
    with torch.no_grad():
        for layer_idx in range(len(model.blocks)):
            num_heads = model.blocks[layer_idx]["attn"].num_heads
            total_count += num_heads
            
            for head_idx in range(num_heads):
                if model.blocks[layer_idx]["attn"].gate[head_idx].item() > 0.5:
                    active_count += 1
    
    return active_count, total_count


def train_epoch(model, train_loader, optimizer, scheduler, device, args):
    """
    Train the model for one epoch.
    
    Args:
        model: The adaptive transformer model
        train_loader: DataLoader with training data
        optimizer: Optimizer for model parameters
        scheduler: Learning rate scheduler
        device: Device to use for training
        args: Command line arguments
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    epoch_loss = 0
    step = 0
    
    metrics = {
        "loss": [],
        "active_heads": []
    }
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Record metrics
        epoch_loss += loss.item()
        metrics["loss"].append(loss.item())
        
        # Record active heads
        active_heads, total_heads = count_active_heads(model)
        metrics["active_heads"].append(active_heads)
        
        step += 1
    
    # Calculate average loss
    avg_loss = epoch_loss / len(train_loader)
    print(f"Average training loss: {avg_loss:.4f}")
    
    return metrics


def evaluate(model, eval_loader, device):
    """
    Evaluate the model on the validation set.
    
    Args:
        model: The adaptive transformer model
        eval_loader: DataLoader with evaluation data
        device: Device to use for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    eval_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            eval_loss += outputs.loss.item()
    
    # Calculate average loss and perplexity
    avg_loss = eval_loss / len(eval_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"Evaluation loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }


def generate_samples(model, tokenizer, device, prompts, max_length=100):
    """
    Generate text samples from the model.
    
    Args:
        model: The adaptive transformer model
        tokenizer: Tokenizer for encoding/decoding text
        device: Device to use for generation
        prompts: List of prompts to generate from
        max_length: Maximum length of generated text
        
    Returns:
        List of generated texts
    """
    model.eval()
    generated_texts = []
    
    for prompt in prompts:
        # Generate text
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            temperature=0.7,
            device=device
        )
        
        generated_texts.append(generated)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}\n")
    
    return generated_texts


def visualize_growth_process(metrics, output_dir):
    """
    Create visualizations of the progressive growth process.
    
    Args:
        metrics: Dictionary with training metrics
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for plotting
    epochs = metrics["epochs"]
    train_loss = metrics["train_loss"]
    eval_loss = metrics["eval_loss"]
    perplexity = metrics["perplexity"]
    active_heads = metrics["active_heads"]
    
    # 1. Plot training and evaluation loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Training Loss", marker="o")
    plt.plot(epochs, eval_loss, label="Evaluation Loss", marker="s")
    plt.title("Loss During Progressive Growth", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "progressive_growth_loss.png"), dpi=300)
    plt.close()
    
    # 2. Plot perplexity
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, perplexity, label="Perplexity", marker="o", color="green")
    plt.title("Perplexity During Progressive Growth", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Perplexity", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "progressive_growth_perplexity.png"), dpi=300)
    plt.close()
    
    # 3. Plot active heads
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, active_heads, label="Active Heads", marker="o", color="purple")
    plt.title("Active Attention Heads During Progressive Growth", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Number of Active Heads", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "progressive_growth_active_heads.png"), dpi=300)
    plt.close()
    
    # 4. Combined plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot loss on left axis
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14, color="blue")
    ax1.plot(epochs, train_loss, label="Training Loss", marker="o", color="blue", alpha=0.7)
    ax1.plot(epochs, eval_loss, label="Evaluation Loss", marker="s", color="green", alpha=0.7)
    ax1.tick_params(axis="y", labelcolor="blue")
    
    # Create second y-axis for active heads
    ax2 = ax1.twinx()
    ax2.set_ylabel("Active Heads", fontsize=14, color="purple")
    ax2.plot(epochs, active_heads, label="Active Heads", marker="d", color="purple", linestyle="--", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="purple")
    
    # Add legend with combined handles from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=12)
    
    plt.title("Progressive Growth: Loss and Active Heads", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "progressive_growth_combined.png"), dpi=300)
    plt.close()
    
    # 5. Create heat map of gate activities at different growth stages
    if "gate_snapshots" in metrics:
        snapshots = metrics["gate_snapshots"]
        num_snapshots = len(snapshots)
        
        # Get dimensions from first snapshot
        if num_snapshots > 0:
            num_layers, num_heads = snapshots[0].shape
            
            fig, axes = plt.subplots(1, num_snapshots, figsize=(4 * num_snapshots, 6))
            if num_snapshots == 1:
                axes = [axes]
            
            for i, (gates, epoch) in enumerate(zip(snapshots, epochs)):
                im = axes[i].imshow(gates, cmap="YlOrRd", vmin=0, vmax=1)
                axes[i].set_title(f"Epoch {epoch}", fontsize=14)
                axes[i].set_xlabel("Attention Head", fontsize=12)
                if i == 0:
                    axes[i].set_ylabel("Transformer Layer", fontsize=12)
                
                # Add grid lines
                axes[i].grid(False)
                axes[i].set_xticks(range(num_heads))
                axes[i].set_yticks(range(num_layers))
            
            fig.colorbar(im, ax=axes, label="Gate Value")
            plt.suptitle("Gate Activity Evolution During Progressive Growth", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir, "progressive_growth_gate_evolution.png"), dpi=300)
            plt.close()
    
    # 6. Create summary table
    summary_data = {
        "Epoch": epochs,
        "Active Heads": active_heads,
        "Active Heads (%)": [h / max(active_heads) * 100 for h in active_heads],
        "Training Loss": train_loss,
        "Eval Loss": eval_loss,
        "Perplexity": perplexity
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "progressive_growth_summary.csv"), index=False)
    
    # 7. Create text summary
    with open(os.path.join(output_dir, "progressive_growth_summary.txt"), "w") as f:
        f.write("Progressive Growth Experiment Summary\n")
        f.write("===================================\n\n")
        f.write(f"Starting Active Heads: {active_heads[0]}\n")
        f.write(f"Final Active Heads: {active_heads[-1]}\n")
        f.write(f"Relative Growth: {active_heads[-1] / active_heads[0]:.2f}x\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"- Initial Perplexity: {perplexity[0]:.2f}\n")
        f.write(f"- Final Perplexity: {perplexity[-1]:.2f}\n")
        f.write(f"- Relative Improvement: {perplexity[0] / perplexity[-1]:.2f}x\n\n")
        
        f.write("Observations:\n")
        if perplexity[-1] < perplexity[0]:
            f.write("- Performance improved with progressive growth\n")
        else:
            f.write("- Performance did not improve significantly\n")
        
        if active_heads[-1] / active_heads[0] > 1.5:
            f.write("- Substantial head growth occurred\n")
        else:
            f.write("- Limited head growth occurred\n")
        
        # Loss trend analysis
        loss_trend = eval_loss[-1] - eval_loss[0]
        if loss_trend < -0.5:
            f.write("- Strong downward trend in loss (significant improvement)\n")
        elif loss_trend < -0.1:
            f.write("- Moderate downward trend in loss (good improvement)\n")
        elif loss_trend < 0:
            f.write("- Slight downward trend in loss (minor improvement)\n")
        else:
            f.write("- No clear improvement trend in loss\n")


def visualize_gate_activity(model, output_dir):
    """
    Visualize the current gate activity across layers and heads.
    
    Args:
        model: The adaptive transformer model
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    
    # Create matrix of gate values
    gate_values = torch.zeros(num_layers, num_heads)
    with torch.no_grad():
        for l in range(num_layers):
            for h in range(num_heads):
                gate_values[l, h] = model.blocks[l]["attn"].gate[h].item()
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(gate_values.numpy(), cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(label="Gate Value")
    plt.title("Attention Head Gate Activity", fontsize=16)
    plt.xlabel("Attention Head", fontsize=14)
    plt.ylabel("Transformer Layer", fontsize=14)
    
    # Add grid lines
    plt.grid(False)
    plt.xticks(range(num_heads))
    plt.yticks(range(num_layers))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gate_activity.png"), dpi=300)
    plt.close()
    
    return gate_values.numpy()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.drive_path:
        output_dir = os.path.join(args.drive_path, args.output_dir, f"{args.dataset}_{timestamp}")
    else:
        output_dir = os.path.join(args.output_dir, f"{args.dataset}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load baseline model
    baseline_model = load_baseline_model(args.model_name, device)
    
    # Create adaptive model
    model = load_adaptive_model(args.model_name, baseline_model, device)
    
    # Apply initial pruning
    model = apply_initial_pruning(
        model, args.initial_strategy, args.initial_pruning, device
    )
    
    # Setup controller manager for U-Net connections
    controller_manager = ControllerManager(model=model)
    
    # Enable U-Net skip connections if requested
    if args.enable_unet:
        print("Enabling U-Net skip connections...")
        controller_manager.enable_unet_connections(
            enable=True, connection_scale=args.connection_scale
        )
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    train_dataset, eval_dataset = load_dataset(
        args.dataset, tokenizer, args.max_length
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Setup learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Tracking metrics
    all_metrics = {
        "epochs": [],
        "train_loss": [],
        "eval_loss": [],
        "perplexity": [],
        "active_heads": [],
        "gate_snapshots": []
    }
    
    # Get the initial growth order (will be updated each epoch)
    growth_order = get_head_growth_order(model, args.growth_strategy, train_loader, device)
    
    # Get the initial active head count
    initial_active_heads, total_heads = count_active_heads(model)
    print(f"Starting with {initial_active_heads} of {total_heads} heads active ({initial_active_heads/total_heads:.1%})")
    
    # Take a snapshot of initial gate activity
    initial_gate_snapshot = visualize_gate_activity(model, output_dir)
    all_metrics["gate_snapshots"].append(initial_gate_snapshot)
    
    # Test prompts for generation
    test_prompts = [
        "Once upon a time in a land far away,",
        "The future of artificial intelligence depends on",
        "In the midst of winter, I found there was, within me,"
    ]
    
    # Generate initial samples
    print("\nGenerating initial samples before growth...")
    initial_samples = generate_samples(model, tokenizer, device, test_prompts)
    
    # Calculate target number of heads to reach
    target_active_heads = int(total_heads * (1 - args.target_pruning))
    heads_to_grow_total = max(0, target_active_heads - initial_active_heads)
    
    # Main training and growth loop
    print(f"\nStarting progressive growth training for {args.epochs} epochs...")
    print(f"Target: {target_active_heads} active heads ({target_active_heads/total_heads:.1%})")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Calculate number of heads to grow this epoch
        if epoch < args.epochs - 1:
            # Distribute growth across epochs
            heads_to_grow = int(heads_to_grow_total * args.growth_rate)
        else:
            # Last epoch - grow all remaining heads to reach target
            current_active, _ = count_active_heads(model)
            heads_to_grow = max(0, target_active_heads - current_active)
        
        # Grow heads if needed
        if heads_to_grow > 0 and growth_order:
            grow_attention_heads(model, heads_to_grow, growth_order, device)
            # Update growth order after growing
            growth_order = get_head_growth_order(model, args.growth_strategy, train_loader, device)
        
        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, args)
        
        # Evaluate
        eval_metrics = evaluate(model, eval_loader, device)
        
        # Get current active head count
        active_heads, _ = count_active_heads(model)
        
        # Update metrics
        all_metrics["epochs"].append(epoch + 1)
        all_metrics["train_loss"].append(np.mean(train_metrics["loss"]))
        all_metrics["eval_loss"].append(eval_metrics["loss"])
        all_metrics["perplexity"].append(eval_metrics["perplexity"])
        all_metrics["active_heads"].append(active_heads)
        
        # Take a snapshot of gate activity
        gate_snapshot = visualize_gate_activity(model, os.path.join(output_dir, f"epoch_{epoch+1}"))
        all_metrics["gate_snapshots"].append(gate_snapshot)
        
        # Report progress
        print(f"Active heads: {active_heads}/{total_heads} ({active_heads/total_heads:.1%})")
        
        # Update growth order for the next epoch
        growth_order = get_head_growth_order(model, args.growth_strategy, train_loader, device)
    
    # Final evaluation
    final_eval_metrics = evaluate(model, eval_loader, device)
    
    # Generate final samples
    print("\nGenerating final samples after growth...")
    final_samples = generate_samples(model, tokenizer, device, test_prompts)
    
    # Create visualizations of the growth process
    visualize_growth_process(all_metrics, output_dir)
    
    # Save the model checkpoint
    checkpoint_path = os.path.join(output_dir, "final_model.pt")
    print(f"Saving model to {checkpoint_path}")
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        head_lr_multipliers=None,
        epoch=args.epochs,
        step=len(train_loader) * args.epochs
    )
    
    # Save the generated text samples
    samples_path = os.path.join(output_dir, "sample_generations.txt")
    with open(samples_path, "w") as f:
        f.write("Sample Generations Before and After Progressive Growth\n")
        f.write("=================================================\n\n")
        
        for i, prompt in enumerate(test_prompts):
            f.write(f"Prompt {i+1}: {prompt}\n\n")
            f.write("Before Growth:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{initial_samples[i]}\n\n")
            f.write("After Growth:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{final_samples[i]}\n\n")
            f.write("=" * 80 + "\n\n")
    
    # Print final summary
    print("\nProgressive Growth Training Complete!")
    print(f"Started with {initial_active_heads}/{total_heads} heads ({initial_active_heads/total_heads:.1%})")
    final_active_heads, _ = count_active_heads(model)
    print(f"Ended with {final_active_heads}/{total_heads} heads ({final_active_heads/total_heads:.1%})")
    print(f"Initial perplexity: {all_metrics['perplexity'][0]:.2f}")
    print(f"Final perplexity: {final_eval_metrics['perplexity']:.2f}")
    print(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()