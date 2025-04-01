#!/usr/bin/env python
"""
Direct probe of model internals to identify fundamental issues.
"""

import os
import argparse
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.loaders.loader import load_baseline_model, load_adaptive_model

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model_structure(model, prefix=""):
    """Recursively print model structure with parameter shapes"""
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if hasattr(module, "weight"):
            print(f"{full_name}: {module.__class__.__name__} with shape {module.weight.shape}")
        else:
            print(f"{full_name}: {module.__class__.__name__}")
        
        get_model_structure(module, full_name)

def track_tensor_flow(model, tokenizer, prompt, device):
    """
    Track tensor values through each layer of the model.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Set model to eval mode
    model.eval()
    
    # Get embedding
    with torch.no_grad():
        if hasattr(model, "transformer"):
            # Baseline GPT-2 model
            wte = model.transformer.wte
            wpe = model.transformer.wpe
            position_ids = torch.arange(len(input_ids.input_ids[0]), device=device).unsqueeze(0)
            token_embed = wte(input_ids.input_ids) 
            pos_embed = wpe(position_ids)
            hidden_state = token_embed + pos_embed
            
            # Print initial embeddings
            print("\n=== BASELINE MODEL FLOW ===")
            print(f"Input sequence: {tokenizer.decode(input_ids.input_ids[0])}")
            print(f"Token embeddings: shape={token_embed.shape}, mean={token_embed.mean().item():.4f}, std={token_embed.std().item():.4f}")
            print(f"Position embeddings: shape={pos_embed.shape}, mean={pos_embed.mean().item():.4f}, std={pos_embed.std().item():.4f}")
            print(f"Initial hidden state: shape={hidden_state.shape}, mean={hidden_state.mean().item():.4f}, std={hidden_state.std().item():.4f}")
            
            # Track hidden states through each layer
            for i, block in enumerate(model.transformer.h):
                # Layer norm before attention
                ln1_out = block.ln_1(hidden_state)
                
                # Get attention outputs (shapes depend on whether attn.output is included)
                if hasattr(block.attn, "c_attn"):
                    # Regular GPT-2 attention with combined QKV projection
                    qkv = block.attn.c_attn(ln1_out)
                    # Split into query, key, value
                    qkv_split = qkv.split(hidden_state.shape[-1], dim=2)
                    q, k, v = qkv_split[0], qkv_split[1], qkv_split[2]
                    
                    # Calculate attention scores
                    att_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.size(-1))
                    att_probs = torch.softmax(att_scores, dim=-1)
                    attn_output = torch.matmul(att_probs, v)
                    attn_output = block.attn.c_proj(attn_output)
                else:
                    # Different structure, try to adapt
                    attn_output = block.attn(ln1_out)[0]  # Assuming returns a tuple with output first
                
                # Add residual connection
                hidden_state = hidden_state + attn_output
                
                # Layer norm before MLP
                ln2_out = block.ln_2(hidden_state)
                
                # Apply MLP
                mlp_output = block.mlp(ln2_out)
                
                # Add residual connection
                hidden_state = hidden_state + mlp_output
                
                # Print stats for this layer
                print(f"Layer {i}:")
                print(f"  After attention: mean={hidden_state.mean().item():.4f}, std={hidden_state.std().item():.4f}")
                print(f"  After MLP: mean={hidden_state.mean().item():.4f}, std={hidden_state.std().item():.4f}")
            
            # Final layer norm
            output = model.transformer.ln_f(hidden_state)
            logits = model.lm_head(output)
            print(f"Final output: shape={logits.shape}, mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
            
            # Top 5 predictions for the last token
            last_token_logits = logits[0, -1, :]
            top5 = torch.topk(last_token_logits, 5)
            print("\nTop 5 predictions:")
            for i, (score, idx) in enumerate(zip(top5.values, top5.indices)):
                token = tokenizer.decode([idx])
                print(f"  {i+1}. '{token}' (ID: {idx}) - Score: {score.item():.4f}")
        
        else:
            # Adaptive model - our custom structure
            print("\n=== ADAPTIVE MODEL FLOW ===")
            print(f"Input sequence: {tokenizer.decode(input_ids.input_ids[0])}")
            
            # Get embeddings
            position_ids = torch.arange(len(input_ids.input_ids[0]), device=device).unsqueeze(0)
            token_embed = model.wte(input_ids.input_ids)
            pos_embed = model.wpe(position_ids)
            hidden_state = token_embed + pos_embed
            
            print(f"Token embeddings: shape={token_embed.shape}, mean={token_embed.mean().item():.4f}, std={token_embed.std().item():.4f}")
            print(f"Position embeddings: shape={pos_embed.shape}, mean={pos_embed.mean().item():.4f}, std={pos_embed.std().item():.4f}")
            print(f"Initial hidden state: shape={hidden_state.shape}, mean={hidden_state.mean().item():.4f}, std={hidden_state.std().item():.4f}")
            
            # Track hidden states through each layer
            for i, block in enumerate(model.blocks):
                # Layer norm before attention
                ln1_out = block["ln1"](hidden_state)
                
                # Get attention outputs
                attn_output = block["attn"](ln1_out)
                
                # Add residual connection
                hidden_state = hidden_state + attn_output
                
                # Layer norm before FFN
                ln2_out = block["ln2"](hidden_state)
                
                # Apply FFN
                ffn_output = block["ffn"](ln2_out)
                
                # Add residual connection
                hidden_state = hidden_state + ffn_output
                
                # Print stats for this layer including gate values
                print(f"Layer {i}:")
                print(f"  Gates: min={block['attn'].gate.min().item():.4f}, max={block['attn'].gate.max().item():.4f}, mean={block['attn'].gate.mean().item():.4f}")
                print(f"  After attention: mean={hidden_state.mean().item():.4f}, std={hidden_state.std().item():.4f}")
                print(f"  After FFN: mean={hidden_state.mean().item():.4f}, std={hidden_state.std().item():.4f}")
            
            # Final processing
            final_ln = model.ln_f(hidden_state)
            logits = model.lm_head(final_ln)
            
            # Check if logits are being scaled (a key issue we identified)
            print(f"Final output: shape={logits.shape}, mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
            
            # Top 5 predictions for the last token
            last_token_logits = logits[0, -1, :]
            top5 = torch.topk(last_token_logits, 5)
            print("\nTop 5 predictions:")
            for i, (score, idx) in enumerate(zip(top5.values, top5.indices)):
                token = tokenizer.decode([idx])
                print(f"  {i+1}. '{token}' (ID: {idx}) - Score: {score.item():.4f}")

def main():
    parser = argparse.ArgumentParser(description="Model probing tool for Sentinel-AI")
    parser.add_argument("--model_name", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Text prompt for probing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--show_structure", action="store_true", help="Show model structure details")
    parser.add_argument("--trace_values", action="store_true", help="Trace tensor values through model")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    print(f"Using seed: {args.seed}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    baseline_model = load_baseline_model(args.model_name, device)
    print("Loaded baseline model")
    
    adaptive_model = load_adaptive_model(args.model_name, baseline_model, device)
    print("Loaded adaptive model")
    
    # Print model structure if requested
    if args.show_structure:
        print("\n=== BASELINE MODEL STRUCTURE ===")
        get_model_structure(baseline_model)
        
        print("\n=== ADAPTIVE MODEL STRUCTURE ===")
        get_model_structure(adaptive_model)
    
    # Trace tensor values through the model if requested
    if args.trace_values:
        track_tensor_flow(baseline_model, tokenizer, args.prompt, device)
        track_tensor_flow(adaptive_model, tokenizer, args.prompt, device)

if __name__ == "__main__":
    main()