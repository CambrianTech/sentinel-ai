"""
Compare GPT-2Block and AdaptiveTransformerBlock forward passes step by step.
"""
import sys
sys.path.insert(0, '/Volumes/FlashGordon/cambrian/sentinel-ai')

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from models.loaders.gpt2_loader_clean import load_adaptive_model_gpt_clean

# Load models
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
gpt2_model.eval()
config = AutoConfig.from_pretrained("gpt2")

adaptive_model = load_adaptive_model_gpt_clean("gpt2", gpt2_model, config, device="cpu", quiet=True)

# Get first layers
gpt2_layer = gpt2_model.transformer.h[0]
adaptive_layer = adaptive_model.transformer.blocks[0]

# Test input
x = torch.randn(1, 5, 768)

print("=" * 80)
print("STEP-BY-STEP FORWARD PASS COMPARISON")
print("=" * 80)

with torch.no_grad():
    # GPT-2 forward pass (manually traced)
    print("\n### GPT-2Block forward pass:")
    
    # Step 1: Save residual
    residual_gpt2 = x
    print(f"1. residual = x, shape: {residual_gpt2.shape}")
    
    # Step 2: LayerNorm 1
    x_gpt2 = gpt2_layer.ln_1(x)
    print(f"2. x = ln_1(x), shape: {x_gpt2.shape}, sample: {x_gpt2[0, 0, :3]}")
    
    # Step 3: Attention
    attn_out_gpt2 = gpt2_layer.attn(x_gpt2)[0]
    print(f"3. attn_out = attn(x), shape: {attn_out_gpt2.shape}, sample: {attn_out_gpt2[0, 0, :3]}")
    
    # Step 4: First residual
    x_gpt2 = residual_gpt2 + attn_out_gpt2
    print(f"4. x = residual + attn_out, sample: {x_gpt2[0, 0, :3]}")
    
    # Step 5: Save residual
    residual_gpt2 = x_gpt2
    print(f"5. residual = x")
    
    # Step 6: LayerNorm 2
    x_gpt2 = gpt2_layer.ln_2(x_gpt2)
    print(f"6. x = ln_2(x), sample: {x_gpt2[0, 0, :3]}")
    
    # Step 7: MLP
    mlp_out_gpt2 = gpt2_layer.mlp(x_gpt2)
    print(f"7. mlp_out = mlp(x), sample: {mlp_out_gpt2[0, 0, :3]}")
    
    # Step 8: Second residual
    x_gpt2 = residual_gpt2 + mlp_out_gpt2
    print(f"8. x = residual + mlp_out, sample: {x_gpt2[0, 0, :3]}")
    
    print(f"\nFinal GPT-2 output sample: {x_gpt2[0, 0, :5]}")
    
    # Adaptive forward pass (manually traced)
    print("\n" + "=" * 80)
    print("### AdaptiveTransformerBlock forward pass:")
    
    # Reset input
    x_adaptive = torch.randn(1, 5, 768)
    x_adaptive.copy_(x)  # Use same input
    
    # Step 1: Save residual
    residual_adaptive = x_adaptive
    print(f"1. residual = hidden_states, shape: {residual_adaptive.shape}")
    
    # Step 2: LayerNorm 1 (prenorm=True)
    x_adaptive = adaptive_layer.norm1(x_adaptive)
    print(f"2. x = norm1(x), sample: {x_adaptive[0, 0, :3]}")
    
    # Step 3: Attention
    attn_out_adaptive = adaptive_layer.attn(x_adaptive)
    print(f"3. attn_out = attn(x), sample: {attn_out_adaptive[0, 0, :3]}")
    
    # Step 4: Dropout (disabled in eval mode but still in graph)
    attn_out_adaptive = adaptive_layer.dropout1(attn_out_adaptive)
    print(f"4. attn_out = dropout1(attn_out), sample: {attn_out_adaptive[0, 0, :3]}")
    
    # Step 5: First residual
    x_adaptive = residual_adaptive + attn_out_adaptive
    print(f"5. x = residual + attn_out, sample: {x_adaptive[0, 0, :3]}")
    
    # Step 6: Save residual
    residual_adaptive = x_adaptive
    print(f"6. residual = x")
    
    # Step 7: LayerNorm 2
    x_adaptive = adaptive_layer.norm2(x_adaptive)
    print(f"7. x = norm2(x), sample: {x_adaptive[0, 0, :3]}")
    
    # Step 8: FFN
    ffn_out_adaptive = adaptive_layer.ffn(x_adaptive)
    print(f"8. ffn_out = ffn(x), sample: {ffn_out_adaptive[0, 0, :3]}")
    
    # Step 9: Second residual
    x_adaptive = residual_adaptive + ffn_out_adaptive
    print(f"9. x = residual + ffn_out, sample: {x_adaptive[0, 0, :3]}")
    
    print(f"\nFinal Adaptive output sample: {x_adaptive[0, 0, :5]}")
    
    print("\n" + "=" * 80)
    print("### COMPARISON:")
    print(f"Max difference: {torch.abs(x_gpt2 - x_adaptive).max().item():.6e}")
    print(f"Mean difference: {torch.abs(x_gpt2 - x_adaptive).mean().item():.6e}")
