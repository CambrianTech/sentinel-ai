#!/usr/bin/env python
"""
Run inference with the Adaptive Transformer model.

This script demonstrates text generation using the adaptive transformer model
with sentinel gates. It supports comparing generation between the baseline model
and the adaptive model to examine differences in outputs and efficiency.

Usage:
    python scripts/inference.py --model_path /path/to/checkpoint.pth \
                              --prompts_file prompts.txt \
                              --output_file generation_results.txt \
                              --compare_baseline

Features:
- Text generation with adaptive transformer model
- Comparison with baseline model
- Visualization of attention patterns
- Tracking of gate values during generation
- Analysis of head utilization
"""

import os
import argparse
import torch
import json
import time
from transformers import AutoTokenizer
from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.checkpoint import load_checkpoint
from utils.generation_wrapper import GenerationWrapper


def load_prompts(prompts_file=None):
    """
    Load prompts from a file or use default prompts.
    
    Args:
        prompts_file: Path to file containing prompts, one per line
        
    Returns:
        List of prompts
    """
    default_prompts = [
        "Once upon a time in a land far away",
        "The scientist made a discovery that would change",
        "The most important thing to remember about artificial intelligence is",
        "When I look back on my life, I realize",
    ]
    
    if not prompts_file or not os.path.exists(prompts_file):
        print(f"No prompts file provided or file not found. Using default prompts.")
        return default_prompts
    
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    if not prompts:
        print(f"No prompts found in {prompts_file}. Using default prompts.")
        return default_prompts
    
    return prompts


def get_head_utilization(model):
    """
    Calculate and return head utilization statistics.
    
    Args:
        model: The adaptive transformer model
        
    Returns:
        Dictionary with head utilization statistics
    """
    head_stats = {}
    
    # Calculate utilization per layer
    for layer_idx, block in enumerate(model.blocks):
        attn_module = block["attn"]
        gate_values = attn_module.gate.detach().cpu().numpy()
        
        # Count active heads (gate > 0.1)
        active_heads = sum(g > 0.1 for g in gate_values)
        total_heads = attn_module.num_heads
        
        head_stats[f"layer_{layer_idx}"] = {
            "active_heads": int(active_heads),
            "total_heads": total_heads,
            "utilization_percent": float(active_heads / total_heads * 100),
            "gate_values": gate_values.tolist()
        }
    
    # Calculate overall statistics
    total_active = sum(layer["active_heads"] for layer in head_stats.values())
    total_heads = sum(layer["total_heads"] for layer in head_stats.values())
    
    head_stats["overall"] = {
        "active_heads": total_active,
        "total_heads": total_heads,
        "utilization_percent": float(total_active / total_heads * 100),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    return head_stats


def main():
    parser = argparse.ArgumentParser(description="Run inference with Adaptive Transformer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--prompts_file", type=str, default=None, help="File containing prompts")
    parser.add_argument("--output_file", type=str, default="generation_results.txt", 
                      help="Output file for generated text")
    parser.add_argument("--compare_baseline", action="store_true",
                      help="Compare generation with baseline model")
    parser.add_argument("--visualize_attention", action="store_true",
                      help="Visualize attention patterns during generation")
    parser.add_argument("--track_gate_values", action="store_true",
                      help="Track gate values during generation")
    parser.add_argument("--max_length", type=int, default=100,
                      help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8,
                      help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                      help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95,
                      help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--num_sequences", type=int, default=1,
                      help="Number of sequences to generate per prompt")
    parser.add_argument("--analysis_file", type=str, default="head_utilization.json",
                      help="File to save head utilization analysis")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                      help="Device to use (defaults to CUDA if available)")
    args = parser.parse_args()
    
    # Determine device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    baseline_model = load_baseline_model(args.model_name, device)
    model = load_adaptive_model(args.model_name, baseline_model, device)
    
    # Load checkpoint
    if os.path.exists(args.model_path):
        optimizer = torch.optim.AdamW(model.parameters())
        head_lr_multipliers = {}
        model, _, _, _, _ = load_checkpoint(model, optimizer, head_lr_multipliers, args.model_path, device)
        print(f"Loaded checkpoint from {args.model_path}")
    
    # Load prompts
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts for generation")
    
    # Create generation wrappers
    adaptive_wrapper = GenerationWrapper(model=model, tokenizer=tokenizer, device=device)
    
    baseline_wrapper = None
    if args.compare_baseline:
        baseline_wrapper = GenerationWrapper(model_name=args.model_name, device=device)
    
    # Generation parameters
    generation_params = {
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": True,
        "num_return_sequences": args.num_sequences,
        "visualize_attention": args.visualize_attention,
        "track_gate_values": args.track_gate_values
    }
    
    # Run generation with adaptive model
    print("\nGenerating text with adaptive model...")
    start_time = time.time()
    adaptive_results = adaptive_wrapper.run_inference(
        prompts, output_file=f"adaptive_{args.output_file}", **generation_params
    )
    adaptive_time = time.time() - start_time
    
    # Run generation with baseline model if requested
    baseline_results = None
    baseline_time = None
    if args.compare_baseline:
        print("\nGenerating text with baseline model...")
        start_time = time.time()
        baseline_results = baseline_wrapper.run_inference(
            prompts, output_file=f"baseline_{args.output_file}", **generation_params
        )
        baseline_time = time.time() - start_time
    
    # Save combined results if comparing
    if args.compare_baseline:
        with open(args.output_file, "w") as f:
            for prompt in prompts:
                f.write(f"Prompt: {prompt}\n\n")
                
                f.write("Adaptive Model:\n")
                for i, text in enumerate(adaptive_results[prompt]):
                    f.write(f"  Generation {i+1}:\n{text}\n\n")
                
                f.write("Baseline Model:\n")
                for i, text in enumerate(baseline_results[prompt]):
                    f.write(f"  Generation {i+1}:\n{text}\n\n")
                
                f.write("-" * 80 + "\n\n")
    
    # Analyze head utilization
    head_stats = get_head_utilization(model)
    
    # Save analysis if requested
    if args.analysis_file:
        # Add generation times to the analysis
        analysis = {
            "head_utilization": head_stats,
            "generation_performance": {
                "adaptive_time_seconds": adaptive_time,
                "adaptive_tokens_per_second": sum(len(prompt.split()) for prompt in prompts) * args.num_sequences * args.max_length / adaptive_time
            }
        }
        
        if args.compare_baseline:
            analysis["generation_performance"].update({
                "baseline_time_seconds": baseline_time,
                "baseline_tokens_per_second": sum(len(prompt.split()) for prompt in prompts) * args.num_sequences * args.max_length / baseline_time,
                "speedup_percent": (baseline_time - adaptive_time) / baseline_time * 100 if baseline_time > 0 else 0
            })
        
        with open(args.analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 40)
    print("Generation Summary:")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Sequences per prompt: {args.num_sequences}")
    print(f"  Max length: {args.max_length}")
    print(f"  Adaptive model time: {adaptive_time:.2f}s")
    
    if args.compare_baseline:
        print(f"  Baseline model time: {baseline_time:.2f}s")
        speedup = (baseline_time - adaptive_time) / baseline_time * 100
        print(f"  Speedup: {speedup:.2f}%")
    
    print("\nHead Utilization:")
    print(f"  Active heads: {head_stats['overall']['active_heads']} / {head_stats['overall']['total_heads']} "
         f"({head_stats['overall']['utilization_percent']:.2f}%)")
    print(f"  Trainable parameters: {head_stats['overall']['trainable_params']:,}")
    
    print("\nResults saved to:")
    if args.compare_baseline:
        print(f"  - {args.output_file} (combined results)")
        print(f"  - adaptive_{args.output_file}")
        print(f"  - baseline_{args.output_file}")
    else:
        print(f"  - adaptive_{args.output_file}")
    
    if args.analysis_file:
        print(f"  - {args.analysis_file} (head utilization analysis)")
    
    if args.visualize_attention:
        print("  - attention_step_*.png (attention visualizations)")
    
    if args.track_gate_values:
        print("  - gate_dynamics.png (gate value dynamics)")


if __name__ == "__main__":
    main()