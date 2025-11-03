"""Test just the attention module in isolation."""
import sys
sys.path.insert(0, '/Volumes/FlashGordon/cambrian/sentinel-ai')

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from models.loaders.gpt2_loader_clean import transfer_attention_weights
from sentinel.models.adaptive_transformer import AdaptiveTransformerBlock

# Load GPT-2
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
gpt2_model.eval()
config = AutoConfig.from_pretrained("gpt2")

# Get layer 0
gpt2_layer = gpt2_model.transformer.h[0]

# Create adaptive block
adaptive_block = AdaptiveTransformerBlock(
    hidden_size=768,
    num_heads=12,
    intermediate_size=3072,
    prenorm=True,
    dropout_prob=0.1
)
adaptive_block.eval()

# Transfer ONLY attention weights
transfer_attention_weights(gpt2_layer, adaptive_block, config)

# Test input (after layer norm, to isolate attention)
x = torch.randn(1, 5, 768)

print("Testing attention in isolation:")
with torch.no_grad():
    gpt2_attn_out = gpt2_layer.attn(x)[0]
    adaptive_attn_out = adaptive_block.attn(x)
    
    print(f"GPT-2 attention output sample: {gpt2_attn_out[0, 0, :5]}")
    print(f"Adaptive attention output sample: {adaptive_attn_out[0, 0, :5]}")
    print(f"Max difference: {torch.abs(gpt2_attn_out - adaptive_attn_out).max().item():.6e}")
    print(f"Mean difference: {torch.abs(gpt2_attn_out - adaptive_attn_out).mean().item():.6e}")
    
    if torch.abs(gpt2_attn_out - adaptive_attn_out).max().item() < 1e-4:
        print("✅ Attention outputs match!")
    else:
        print("❌ Attention outputs differ!")
        
        # Debug: check gate values
        print(f"\nGate values: {adaptive_block.attn.gate.data}")
        
        # Check if it's a magnitude issue
        gpt2_mag = gpt2_attn_out.abs().mean().item()
        adaptive_mag = adaptive_attn_out.abs().mean().item()
        print(f"GPT-2 magnitude: {gpt2_mag:.6f}")
        print(f"Adaptive magnitude: {adaptive_mag:.6f}")
        print(f"Ratio: {adaptive_mag / gpt2_mag:.6f}")
