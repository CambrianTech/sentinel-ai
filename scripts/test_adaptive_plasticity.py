#!/usr/bin/env python
"""
Test Adaptive Neural Plasticity System

This script runs a comprehensive test of the neural plasticity system that:
1. Tests both entropy and magnitude-based pruning
2. Demonstrates the complete plasticity cycle (prune → fine-tune → analyze)
3. Shows gate activity tracking and entropy changes
4. Visualizes head regrowth patterns
5. Generates detailed reports and visualizations

The script showcases the scientific approach to studying neural networks:
- How transformer models reorganize after pruning
- Which heads regrow during fine-tuning
- How entropy changes after adaptation
- How performance recovers through plasticity
"""

import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Sentinel-AI modules
from sentinel.plasticity.plasticity_loop import PlasticityExperiment, run_plasticity_experiment
from sentinel.utils.viz.heatmaps import (
    plot_entropy_heatmap,
    plot_entropy_deltas_heatmap,
    plot_gate_activity,
    plot_regrowth_heatmap
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test Adaptive Neural Plasticity System")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Model name (default: distilgpt2)")
    parser.add_argument("--output_dir", type=str, default="./output/plasticity_test",
                        help="Directory to save results (default: ./output/plasticity_test)")
    
    # Experiment parameters
    parser.add_argument("--pruning_strategy", type=str, default="entropy",
                        choices=["entropy", "magnitude"],
                        help="Pruning strategy (default: entropy)")
    parser.add_argument("--pruning_level", type=float, default=0.3,
                        help="Pruning level (default: 0.3)")
    parser.add_argument("--training_steps", type=int, default=200,
                        help="Number of fine-tuning steps (default: 200)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4)")
    
    # System parameters
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--show_visualizations", action="store_true",
                        help="Show visualizations (in addition to saving)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    return parser.parse_args()

def get_dataloader_builder(batch_size=4):
    """
    Create a function that returns train and evaluation dataloaders.
    Uses a simple dataset for testing purposes.
    """
    from transformers import AutoTokenizer
    import torch
    
    # Create synthetic data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a world where technology dominates, humans seek connection.",
        "Once upon a time, there lived a wise king who ruled with compassion.",
        "The history of artificial intelligence dates back to ancient myths.",
        "Climate change is affecting ecosystems worldwide, leading to rising sea levels.",
        "The transformer architecture revolutionized natural language processing tasks.",
        "Neural plasticity allows models to adapt their structure during training.",
        "Deep learning models can recognize patterns in complex data.",
        "The attention mechanism focuses on different parts of the input sequence.",
        "Language models predict the next token based on previous context."
    ] * 10  # Repeat to create more samples
    
    def build_dataloaders(model_name="distilgpt2", batch_size=4):
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize
        from torch.utils.data import TensorDataset, DataLoader
        
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        dataset = TensorDataset(input_ids, attention_mask)
        
        # Split into train and eval
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        
        return train_dataloader, eval_dataloader
    
    # Return a function that will create dataloaders with the specified batch size
    return lambda batch_size=batch_size: build_dataloaders(batch_size=batch_size)

def create_visualizations(results_dir, results, show=False):
    """Create visualizations from plasticity experiment results"""
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load data from results files
    try:
        import json
        import torch
        import numpy as np
        
        # Load pre/post entropy data
        with open(os.path.join(results_dir, "pre_entropy.json"), 'r') as f:
            pre_entropy_data = json.load(f)
            # Convert from serialized format back to tensors
            pre_entropy = {int(k): torch.tensor(v) for k, v in pre_entropy_data.items()}
            
        with open(os.path.join(results_dir, "post_entropy.json"), 'r') as f:
            post_entropy_data = json.load(f)
            post_entropy = {int(k): torch.tensor(v) for k, v in post_entropy_data.items()}
            
        with open(os.path.join(results_dir, "entropy_deltas.json"), 'r') as f:
            deltas_data = json.load(f)
            entropy_deltas = {int(k): torch.tensor(v) for k, v in deltas_data.items()}
            
        # Load gate history
        with open(os.path.join(results_dir, "gate_history.json"), 'r') as f:
            gate_history_data = json.load(f)
            gate_history = {}
            for step, layers in gate_history_data.items():
                gate_history[int(step)] = {}
                for layer, gates in layers.items():
                    gate_history[int(step)][int(layer)] = torch.tensor(gates)
                    
        # Load regrowth analysis
        with open(os.path.join(results_dir, "regrowth_analysis.json"), 'r') as f:
            regrowth_data = json.load(f)
            # Convert to the format expected by plot_regrowth_heatmap
            regrowth_analysis = {}
            for key, data in regrowth_data.items():
                layer_idx, head_idx = map(int, key.split('_'))
                regrowth_analysis[(layer_idx, head_idx)] = data
        
        # Create visualizations
        
        # 1. Pre-pruning entropy heatmap
        pre_entropy_fig = plot_entropy_heatmap(
            pre_entropy,
            title="Attention Entropy Before Fine-tuning",
            save_path=os.path.join(viz_dir, "pre_entropy_heatmap.png")
        )
        
        # 2. Post-fine-tuning entropy heatmap
        post_entropy_fig = plot_entropy_heatmap(
            post_entropy,
            title="Attention Entropy After Fine-tuning",
            save_path=os.path.join(viz_dir, "post_entropy_heatmap.png")
        )
        
        # 3. Entropy change heatmap
        delta_entropy_fig = plot_entropy_deltas_heatmap(
            entropy_deltas,
            title="Entropy Change After Fine-tuning",
            save_path=os.path.join(viz_dir, "entropy_deltas_heatmap.png")
        )
        
        # 4. Gate activity for regrown heads
        if regrowth_analysis:
            regrown_heads = list(regrowth_analysis.keys())
            gate_activity_fig = plot_gate_activity(
                gate_history,
                head_indices=regrown_heads,
                title="Gate Activity for Regrown Heads During Fine-tuning",
                save_path=os.path.join(viz_dir, "gate_activity_regrown.png")
            )
            
            # 5. Regrowth heatmap
            regrowth_fig = plot_regrowth_heatmap(
                regrowth_analysis,
                title="Head Regrowth Analysis",
                save_path=os.path.join(viz_dir, "regrowth_heatmap.png")
            )
        else:
            print("No regrown heads detected")
        
        # 6. Create a combined visualization of metrics
        metrics_data = results.get("metrics", {})
        if metrics_data:
            stages = ["baseline", "post_pruning", "final"]
            perplexities = [metrics_data.get(stage, {}).get("perplexity", 0) for stage in stages]
            
            plt.figure(figsize=(10, 6))
            plt.bar(stages, perplexities, color=['green', 'red', 'blue'])
            plt.ylabel('Perplexity (lower is better)')
            plt.title('Model Perplexity Through Plasticity Cycle')
            
            # Add value labels
            for i, v in enumerate(perplexities):
                plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
                
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "perplexity_comparison.png"))
            
            # Calculate recovery rate
            if "post_pruning" in metrics_data and "baseline" in metrics_data and "final" in metrics_data:
                recovery_rate = results.get("recovery_rate", 0.0)
                
                print(f"\nRecovery Analysis:")
                print(f"Baseline perplexity: {metrics_data['baseline']['perplexity']:.2f}")
                print(f"Post-pruning perplexity: {metrics_data['post_pruning']['perplexity']:.2f}")
                print(f"Final perplexity: {metrics_data['final']['perplexity']:.2f}")
                print(f"Recovery rate: {recovery_rate:.2%}")
        
        # Show figures if requested
        if show:
            plt.show()
            
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_name}_{args.pruning_strategy}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Testing Adaptive Neural Plasticity System ===")
    print(f"Model: {args.model_name}")
    print(f"Pruning strategy: {args.pruning_strategy}")
    print(f"Pruning level: {args.pruning_level}")
    print(f"Training steps: {args.training_steps}")
    print(f"Output directory: {output_dir}")
    
    # Create dataloader builder
    dataloader_builder = get_dataloader_builder(batch_size=args.batch_size)
    
    # Run plasticity experiment
    results = run_plasticity_experiment(
        model_name=args.model_name,
        pruning_strategy=args.pruning_strategy,
        prune_ratio=args.pruning_level,
        learning_rate=args.learning_rate,
        adaptive_lr=True,  # Use differential learning rates
        learning_steps=args.training_steps,
        batch_size=args.batch_size,
        dataloader_builder_fn=dataloader_builder,
        device=args.device,
        output_dir=output_dir
    )
    
    # Print success message
    print(f"\nPlasticity experiment completed successfully!")
    print(f"Results saved to: {output_dir}")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    create_visualizations(output_dir, results, show=args.show_visualizations)
    
    # Print summary
    recovery_rate = results.get("recovery_rate", 0.0)
    print(f"\nExperiment Summary:")
    print(f"- Model: {args.model_name}")
    print(f"- Pruning: {args.pruning_strategy} at {args.pruning_level:.2f} level")
    print(f"- Recovery rate: {recovery_rate:.2%}")
    print(f"- Visualizations saved to: {os.path.join(output_dir, 'visualizations')}")

if __name__ == "__main__":
    main()