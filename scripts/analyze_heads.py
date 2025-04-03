#!/usr/bin/env python
"""
Analyze attention head metrics and visualize head activity for a trained model.
This script helps understand which heads are most active/important and can guide
pruning or expansion decisions.

Usage:
    python scripts/analyze_heads.py --model_path /path/to/checkpoint.pth \
                                  --dataset tiny_shakespeare \
                                  --output_dir ./head_analysis/

Features:
- Calculates attention entropy per head
- Measures head importance via ablation
- Computes gradient norms
- Visualizes head metrics
- Analyses head similarity for redundancy
- Recommends heads to prune or layers to expand
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from models.loaders.loader import load_baseline_model, load_adaptive_model
from models.adaptive_transformer import AdaptiveCausalLmWrapper
from custdata.loaders.dataset_loader import load_and_tokenize_dataset
from utils.checkpoint import load_checkpoint
from utils.head_metrics import (
    compute_attention_entropy,
    compute_head_importance,
    compute_gradient_norms,
    analyze_head_clustering,
    visualize_head_metrics,
    visualize_head_similarities,
    recommend_pruning_growth
)
from utils.training import compute_loss


def visualize_gate_values(model, save_path=None):
    """Visualize gate values for all attention heads."""
    gate_values = {}
    
    for layer_idx, block in enumerate(model.blocks):
        attn_module = block["attn"]
        gate_values[layer_idx] = attn_module.gate.detach().cpu()
    
    # Convert to matrix for visualization
    num_layers = len(gate_values)
    num_heads = len(gate_values[0])
    gate_matrix = torch.zeros(num_layers, num_heads)
    
    for layer_idx, gates in gate_values.items():
        gate_matrix[layer_idx] = gates
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(gate_matrix.numpy(), cmap="viridis", aspect="auto")
    plt.colorbar(label="Gate Value")
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    plt.title("Attention Gate Values")
    
    # Add text annotations
    for i in range(num_layers):
        for j in range(num_heads):
            plt.text(j, i, f"{gate_matrix[i, j]:.2f}", 
                    ha="center", va="center", 
                    color="white" if gate_matrix[i, j] < 0.5 else "black")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Gate values visualization saved to {save_path}")
    
    plt.close()


def create_head_report(model, dataloader, device, output_dir):
    """Generate a comprehensive report on head metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Computing attention entropy...")
    entropy_dict = compute_attention_entropy(model, device=device)
    
    print("Computing head importance via ablation...")
    importance_dict = compute_head_importance(model, dataloader, compute_loss, device=device)
    
    # Create optimizer for gradient computation
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    print("Computing gradient norms...")
    grad_norms_dict = compute_gradient_norms(model, dataloader, compute_loss, optimizer, device=device)
    
    print("Analyzing head clustering for redundancy...")
    similarity_dict = analyze_head_clustering(model, dataloader, device=device)
    
    print("Generating visualizations...")
    # Visualize metrics
    metrics_path = os.path.join(output_dir, "head_metrics.png")
    visualize_head_metrics(entropy_dict, importance_dict, grad_norms_dict, save_path=metrics_path)
    
    # Visualize gate values
    gate_path = os.path.join(output_dir, "gate_values.png")
    visualize_gate_values(model, save_path=gate_path)
    
    # Visualize head similarities
    similarity_path = os.path.join(output_dir, "head_similarities.png")
    visualize_head_similarities(similarity_dict, save_path=similarity_path)
    
    # Get pruning and growth recommendations
    prune_candidates, grow_candidates = recommend_pruning_growth(
        entropy_dict, importance_dict, grad_norms_dict)
    
    # Generate text report
    report_path = os.path.join(output_dir, "head_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("Adaptive Transformer Head Analysis Report\n")
        f.write("========================================\n\n")
        
        f.write("Model Statistics:\n")
        f.write(f"  Total layers: {len(model.blocks)}\n")
        f.write(f"  Heads per layer: {model.blocks[0]['attn'].num_heads}\n")
        f.write(f"  Active heads: {sum(float(block['attn'].gate[i]) > 0.2 for block in model.blocks for i in range(block['attn'].num_heads))}\n")
        f.write(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")
        
        f.write("Pruning Recommendations:\n")
        if prune_candidates:
            for layer_idx, head_idx in prune_candidates:
                f.write(f"  Layer {layer_idx}, Head {head_idx}:\n")
                f.write(f"    Entropy: {entropy_dict[layer_idx][head_idx].item():.4f}\n")
                f.write(f"    Importance: {importance_dict[layer_idx][head_idx].item():.4f}\n")
                f.write(f"    Gradient norm: {grad_norms_dict[layer_idx][head_idx].item():.4e}\n")
                f.write(f"    Current gate value: {model.blocks[layer_idx]['attn'].gate[head_idx].item():.4f}\n\n")
        else:
            f.write("  No heads recommended for pruning\n\n")
        
        f.write("Growth Recommendations:\n")
        if grow_candidates:
            for layer_idx in grow_candidates:
                f.write(f"  Layer {layer_idx}:\n")
                avg_entropy = entropy_dict[layer_idx].mean().item()
                avg_importance = importance_dict[layer_idx].mean().item()
                f.write(f"    Average entropy: {avg_entropy:.4f}\n")
                f.write(f"    Average importance: {avg_importance:.4f}\n")
                f.write(f"    Current active heads: {sum(float(model.blocks[layer_idx]['attn'].gate[i]) > 0.2 for i in range(model.blocks[layer_idx]['attn'].num_heads))}\n\n")
        else:
            f.write("  No layers recommended for growth\n\n")
            
    print(f"Analysis report generated at {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Analyze attention head metrics")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./head_analysis/", help="Output directory")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                      help="Device to use (defaults to CUDA if available)")
    parser.add_argument("--implement_recommendations", action="store_true",
                      help="Automatically implement pruning/growth recommendations")
    parser.add_argument("--save_modified_model", type=str, default=None,
                      help="Path to save modified model if recommendations are implemented")
    args = parser.parse_args()
    
    # Determine device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    baseline_model = load_baseline_model(args.model_name, device)
    model = load_adaptive_model(args.model_name, baseline_model, device)
    
    # Load checkpoint if provided
    if args.model_path and os.path.exists(args.model_path):
        optimizer = torch.optim.AdamW(model.parameters())
        head_lr_multipliers = {}
        model, optimizer, head_lr_multipliers, epoch, step = load_checkpoint(
            model, optimizer, head_lr_multipliers, args.model_path, device)
        print(f"Loaded checkpoint from {args.model_path}")
    
    # Load dataset
    train_ids, val_ids = load_and_tokenize_dataset(args.model_name, dataset_name=args.dataset)
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_ids))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
    
    # Create analysis directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate head report
    print("\nGenerating comprehensive head analysis...")
    report_path = create_head_report(model, val_loader, device, args.output_dir)
    
    # Implement recommendations if requested
    if args.implement_recommendations and args.save_modified_model:
        print("\nImplementing recommendations...")
        
        # Recompute metrics
        entropy_dict = compute_attention_entropy(model, device=device)
        importance_dict = compute_head_importance(model, val_loader, compute_loss, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        grad_norms_dict = compute_gradient_norms(model, val_loader, compute_loss, optimizer, device=device)
        
        # Get recommendations
        prune_candidates, grow_candidates = recommend_pruning_growth(
            entropy_dict, importance_dict, grad_norms_dict)
        
        # Implement pruning
        if prune_candidates:
            print(f"Pruning {len(prune_candidates)} heads...")
            for layer_idx, head_idx in prune_candidates:
                # Set gate to zero and freeze parameters
                attn_module = model.blocks[layer_idx]["attn"]
                attn_module.gate.data[head_idx] = 0.0
                
                # Freeze parameters for this head
                for param in attn_module.W_q[head_idx].parameters():
                    param.requires_grad = False
                for param in attn_module.W_k[head_idx].parameters():
                    param.requires_grad = False
                for param in attn_module.W_v[head_idx].parameters():
                    param.requires_grad = False
                for param in attn_module.W_o[head_idx].parameters():
                    param.requires_grad = False
                
                print(f"  Pruned Layer {layer_idx}, Head {head_idx}")
        
        # Implement growth using expand_heads.py functionality
        if grow_candidates and len(grow_candidates) > 0:
            from scripts.expand_heads import expand_attention_heads
            
            print(f"Expanding heads in {len(grow_candidates)} layers...")
            heads_per_layer = 2  # Default value
            model = expand_attention_heads(model, grow_candidates, heads_per_layer)
            
            print(f"  Added {heads_per_layer} heads to each of layers {grow_candidates}")
        
        # Save modified model
        print(f"Saving modified model to {args.save_modified_model}")
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "head_lr_multipliers": head_lr_multipliers,
            "epoch": epoch,
            "train_step": step
        }, args.save_modified_model)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    

if __name__ == "__main__":
    main()