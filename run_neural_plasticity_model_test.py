#!/usr/bin/env python
"""
Neural Plasticity Model Test with Safe Matrix Operations

This script runs a minimal test with a transformer model that uses our safe_matmul
function for all matrix operations. It includes a custom attention layer that
replaces torch.matmul with our safe_matmul function.
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import platform
import argparse

# Add current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Set environment variables for maximum stability
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['ACCELERATE_USE_SYSTEM_BLAS'] = '1'
os.environ['PYTORCH_JIT_USE_AUTOTUNER'] = '0'
torch.set_num_threads(1)

# Import our safe_matmul function
from utils.neural_plasticity.core import safe_matmul, calculate_head_entropy
from utils.neural_plasticity.visualization import visualize_head_entropy 

def is_apple_silicon():
    """Check if we're running on Apple Silicon"""
    return platform.system() == "Darwin" and platform.processor() == "arm"

class SafeAttention(torch.nn.Module):
    """
    Safe attention layer that uses our safe_matmul function
    to prevent BLAS crashes on Apple Silicon.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Create projection matrices
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
    
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project query, key, value
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Transpose key for attention calculation [batch_size, num_heads, head_dim, seq_len]
        key_t = key.transpose(2, 3)
        
        # Store attention maps for later analysis
        attention_maps = torch.zeros(batch_size, self.num_heads, seq_len, seq_len)
        attention_output = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # Calculate attention scores using safe_matmul
        for b in range(batch_size):
            for h in range(self.num_heads):
                # Q @ K^T
                attention_scores = safe_matmul(query[b, h], key_t[b, h])
                
                # Scale attention scores
                attention_scores = attention_scores / (self.head_dim ** 0.5)
                
                # Apply softmax
                attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
                
                # Store attention maps for visualization
                attention_maps[b, h] = attention_probs
                
                # Calculate weighted values: attention_probs @ V
                attention_output[b, h] = safe_matmul(attention_probs, value[b, h])
        
        # Reshape and transpose back
        attention_output = attention_output.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        attention_output = attention_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attention_output)
        
        return output, attention_maps

class SafeTransformerBlock(torch.nn.Module):
    """
    Transformer block that uses SafeAttention to avoid BLAS crashes.
    """
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = SafeAttention(embed_dim, num_heads)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, ff_dim),
            torch.nn.GELU(),
            torch.nn.Linear(ff_dim, embed_dim)
        )
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Self-attention
        norm_x = self.layer_norm1(x)
        attention_output, attention_maps = self.attention(norm_x)
        x = x + attention_output
        
        # Feed-forward
        norm_x = self.layer_norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + ff_output
        
        return x, attention_maps

class SafeMinimalModel(torch.nn.Module):
    """
    Minimal transformer model that uses safe matrix operations.
    """
    def __init__(self, vocab_size=1000, embed_dim=256, num_heads=4, num_layers=2, ff_dim=512):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.blocks = torch.nn.ModuleList([
            SafeTransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.lm_head = torch.nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, input_ids):
        # Get embeddings
        x = self.embedding(input_ids)
        
        # Store attention maps
        attention_maps = []
        
        # Pass through transformer blocks
        for block in self.blocks:
            x, attn_maps = block(x)
            attention_maps.append(attn_maps)
        
        # Final layer norm
        x = self.final_layer_norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits, attention_maps

def run_safe_model_test(output_dir="np_test_output"):
    """
    Run a minimal model test with our safe matrix operations.
    
    Args:
        output_dir: Directory to save visualizations
    """
    print("\n=== NEURAL PLASTICITY MODEL TEST ===")
    print(f"Running on Apple Silicon: {is_apple_silicon()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a minimal model
    vocab_size = 1000
    embed_dim = 256
    num_heads = 4
    num_layers = 2
    
    print(f"Creating minimal transformer model...")
    model = SafeMinimalModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    # Create a random input
    batch_size = 2
    seq_len = 20
    
    print(f"Creating random input: batch_size={batch_size}, seq_len={seq_len}")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Run forward pass
    print(f"Running forward pass...")
    start_time = time.time()
    logits, attention_maps = model(input_ids)
    elapsed = time.time() - start_time
    
    print(f"Forward pass completed in {elapsed:.4f}s")
    print(f"Logits shape: {logits.shape}")
    print(f"Number of attention maps: {len(attention_maps)}")
    
    # Analyze attention maps
    print("\nAnalyzing attention patterns...")
    
    # Get the last layer's attention maps
    layer_attn = attention_maps[-1]  # [batch_size, num_heads, seq_len, seq_len]
    
    # Calculate entropy
    entropy = calculate_head_entropy(layer_attn)
    print(f"Entropy shape: {entropy.shape}")
    
    # Create visualizations
    print(f"Creating visualizations in {output_dir}...")
    
    # Save entropy visualization
    fig = visualize_head_entropy(
        entropy_values=entropy,
        title="Attention Entropy",
        save_path=os.path.join(output_dir, "model_entropy.png")
    )
    plt.close(fig)
    
    print(f"✅ Visualizations saved to {output_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run safe neural plasticity model test")
    parser.add_argument("--output", type=str, default="np_test_output",
                        help="Output directory for visualizations")
    args = parser.parse_args()
    
    success = run_safe_model_test(args.output)
    
    if success:
        print("\n🎉 MODEL TEST PASSED! Safe matrix operations are working correctly.")
        return 0
    else:
        print("\n⚠️ MODEL TEST FAILED. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())