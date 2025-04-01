#!/usr/bin/env python
"""
Debug comparison script for Sentinel-AI.

This script helps debug the adaptive model by directly comparing baseline and adaptive model outputs,
hidden states, and attention patterns. It's designed to isolate exactly where divergence occurs.
"""

import os
import argparse
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.generation_wrapper import GenerationWrapper
import matplotlib.pyplot as plt

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compare_model_outputs(baseline_model, adaptive_model, tokenizer, prompt, device):
    """Compare outputs between baseline and adaptive models for debugging"""
    # Tokenize the input
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Set both models to eval mode
    baseline_model.eval()
    adaptive_model.eval()
    
    # Forward pass through both models
    with torch.no_grad():
        baseline_outputs = baseline_model(**input_ids)
        adaptive_outputs = adaptive_model(input_ids.input_ids)
        
        # Convert adaptive output to same format if needed
        if isinstance(adaptive_outputs, torch.Tensor):
            adaptive_logits = adaptive_outputs
        else:
            adaptive_logits = adaptive_outputs.logits
            
        baseline_logits = baseline_outputs.logits
    
    # Compare logit distributions
    print(f"\n==== LOGIT COMPARISON ====")
    
    # Get statistics
    baseline_mean = baseline_logits.mean().item()
    adaptive_mean = adaptive_logits.mean().item()
    baseline_std = baseline_logits.std().item()
    adaptive_std = adaptive_logits.std().item()
    
    print(f"Baseline logits - Mean: {baseline_mean:.4f}, Std: {baseline_std:.4f}")
    print(f"Adaptive logits - Mean: {adaptive_mean:.4f}, Std: {adaptive_std:.4f}")
    
    # Get difference
    abs_diff = (baseline_logits - adaptive_logits).abs()
    mean_diff = abs_diff.mean().item()
    max_diff = abs_diff.max().item()
    
    print(f"Difference - Mean: {mean_diff:.4f}, Max: {max_diff:.4f}")
    
    # Compare top predictions
    top_k = 5
    
    # Get the last token position predictions
    baseline_last = baseline_logits[:, -1, :]
    adaptive_last = adaptive_logits[:, -1, :]
    
    # Get top-k tokens
    baseline_topk = baseline_last.topk(top_k)
    adaptive_topk = adaptive_last.topk(top_k)
    
    print(f"\n==== TOP {top_k} TOKENS COMPARISON ====")
    print("Baseline model:")
    for i, (score, token_id) in enumerate(zip(baseline_topk.values[0], baseline_topk.indices[0])):
        token = tokenizer.decode([token_id])
        print(f" {i+1}. '{token}' (ID: {token_id}) - Score: {score:.4f}")
        
    print("\nAdaptive model:")
    for i, (score, token_id) in enumerate(zip(adaptive_topk.values[0], adaptive_topk.indices[0])):
        token = tokenizer.decode([token_id])
        print(f" {i+1}. '{token}' (ID: {token_id}) - Score: {score:.4f}")
    
    # Generate text from both models
    print("\n==== GENERATED TEXT COMPARISON ====")
    
    # Generation parameters
    gen_params = {
        "max_length": len(input_ids.input_ids[0]) + 30,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "do_sample": True,
        "num_return_sequences": 1
    }
    
    # Generate from baseline
    baseline_gen = baseline_model.generate(
        input_ids.input_ids,
        **gen_params
    )
    baseline_text = tokenizer.decode(baseline_gen[0], skip_special_tokens=True)
    
    # Create wrapper for adaptive model
    adaptive_wrapper = GenerationWrapper(model=adaptive_model, tokenizer=tokenizer, device=device)
    
    # Generate from adaptive model
    adaptive_gen = adaptive_wrapper.generate_text(
        prompt,
        **gen_params
    )
    adaptive_text = adaptive_gen[0]
    
    print(f"Prompt: {prompt}")
    print(f"\nBaseline generated: {baseline_text}")
    print(f"\nAdaptive generated: {adaptive_text}")
    
    return baseline_logits, adaptive_logits

def compare_gate_distributions(adaptive_model):
    """Analyze distribution of gate values"""
    print("\n==== GATE VALUE ANALYSIS ====")
    
    all_gates = []
    
    # Collect gate values from all layers
    for layer_idx, block in enumerate(adaptive_model.blocks):
        gates = block["attn"].gate.detach().cpu().numpy()
        all_gates.append(gates)
        
        # Print statistics for this layer
        mean = gates.mean()
        std = gates.std()
        min_val = gates.min()
        max_val = gates.max()
        
        print(f"Layer {layer_idx}: Mean={mean:.4f}, Std={std:.4f}, Min={min_val:.4f}, Max={max_val:.4f}")
    
    # Plot histogram of all gate values
    all_gates = np.concatenate(all_gates)
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=20)
    plt.title("Distribution of Gate Values")
    plt.xlabel("Gate Value")
    plt.ylabel("Count")
    plt.savefig("gate_distribution.png")
    print(f"Gate distribution histogram saved to gate_distribution.png")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug comparison tool for Sentinel-AI models",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--model_name", type=str, default=os.getenv("MODEL_NAME", "gpt2"),
                        help="HuggingFace model name.")
    parser.add_argument("--prompt", type=str, default="The future of AI is",
                        help="Prompt text for comparison.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                        help="Compute device to use: 'cpu' or 'cuda' (default: auto-detect).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--disable_gates", action="store_true", 
                        help="Set all gate values to 1.0 for testing.")
    parser.add_argument("--disable_logit_scaling", action="store_true", 
                        help="Disable logit scaling in adaptive model.")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducible results
    set_seed(args.seed)
    print(f"🎲 Random seed set to: {args.seed}")
        
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"🚀 Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the baseline model
    baseline_model = load_baseline_model(args.model_name, device)
    print("⚙️  Loaded baseline model")
    
    # Load the adaptive model
    adaptive_model = load_adaptive_model(args.model_name, baseline_model, device)
    print("⚙️  Loaded adaptive model")
    
    # Apply any debugging modifications
    if args.disable_gates:
        print("⚠️  Setting all gate values to 1.0")
        with torch.no_grad():
            for block in adaptive_model.blocks:
                block["attn"].gate.fill_(1.0)
                
    # Compare model outputs
    baseline_logits, adaptive_logits = compare_model_outputs(
        baseline_model, adaptive_model, tokenizer, args.prompt, device)
    
    # Analyze gate distributions
    compare_gate_distributions(adaptive_model)
    
    print("\n✅ Debug comparison complete")

if __name__ == "__main__":
    main()