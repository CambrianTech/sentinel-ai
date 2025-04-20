#!/usr/bin/env python
"""
Run Neural Plasticity Demo with Comprehensive Coverage

This script executes all the core functionality from the NeuralPlasticityDemo notebook
in a controlled environment, ensuring comprehensive coverage across key components.

Usage:
  source .venv/bin/activate  # Activate virtual environment
  python scripts/run_neural_plasticity_demo.py
"""

import os
import sys
import time
import platform
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure matplotlib for non-interactive mode
matplotlib.use('Agg')

# ===== ENVIRONMENT SETUP =====
print("⏱️ Starting Neural Plasticity Demo execution...")
start_time = time.time()

# Detect Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"

# Apply environment-specific optimizations
if IS_APPLE_SILICON:
    print("🍎 Apple Silicon detected - applying optimizations")
    # Force PyTorch to use CPU and single-threading for stability
    torch.set_num_threads(1)
    
    # Force single-threaded BLAS operations
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Disable BLAS threading at pytorch level
    try:
        torch.backends.openmp.is_available = lambda: False
        torch.backends.mkldnn.enabled = False
        torch.set_deterministic(True)
    except (AttributeError, RuntimeError) as e:
        print(f"⚠️ Could not set all PyTorch safeguards: {e}")
    
    # Force use of safer BLAS implementation
    os.environ["ACCELERATE_USE_SYSTEM_BLAS"] = "1"
    os.environ["PYTORCH_JIT_USE_AUTOTUNER"] = "0"

# Check GPU availability (skip on Apple Silicon)
if not IS_APPLE_SILICON:
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        device = torch.device("cuda")
        print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU (no GPU detected)")
else:
    device = torch.device("cpu")
    print("💻 Using CPU (forced on Apple Silicon)")

# ===== SAFE TENSOR OPERATIONS =====
def safe_matmul(a, b):
    """Safely perform matrix multiplication on any platform."""
    if IS_APPLE_SILICON:
        # Ensure tensors are on CPU
        if a.is_cuda:
            a = a.cpu()
        if b.is_cuda:
            b = b.cpu()
        
        # Ensure tensors are contiguous and float32
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
            
        # Try multiple approaches for safety
        try:
            # Try numpy first (bypasses BLAS issues)
            a_np = a.detach().cpu().numpy()
            b_np = b.detach().cpu().numpy()
            result_np = np.matmul(a_np, b_np)
            return torch.tensor(result_np, device='cpu')
        except Exception:
            # Fall back to careful PyTorch matmul
            with torch.no_grad():
                result = torch.matmul(a.detach(), b.detach())
                # Check for NaN/Inf values
                if torch.isnan(result).any() or torch.isinf(result).any():
                    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
                return result
    else:
        # Standard matmul for non-Apple Silicon
        return torch.matmul(a, b)

def calculate_head_entropy(attention_maps, eps=1e-6):
    """Calculate entropy of attention patterns."""
    if not isinstance(attention_maps, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(attention_maps)}")
    
    # Special handling for different environments
    if attention_maps.is_cuda and device.type == "cuda":
        # For GPUs, ensure tensor is on GPU
        if not attention_maps.is_cuda:
            attention_maps = attention_maps.to(device)
    elif IS_APPLE_SILICON:
        # Apple Silicon workaround
        if attention_maps.is_cuda:
            attention_maps = attention_maps.detach().cpu()
        if attention_maps.requires_grad:
            attention_maps = attention_maps.detach()
    
    # Calculate entropy safely
    attn_probs = torch.where(
        torch.isfinite(attention_maps),
        attention_maps,
        torch.ones_like(attention_maps) * eps
    )
    attn_probs = attn_probs.clamp(min=eps)
    
    # Normalize to ensure it's a proper probability distribution
    attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)
    
    # Calculate entropy: -sum(p * log(p))
    if attn_probs.dtype != torch.float32:
        attn_probs = attn_probs.to(torch.float32)
    entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)
    
    # Average over batch and sequence length dimensions if present
    dims_to_reduce = list(range(entropy.dim() - 2))
    if dims_to_reduce:
        entropy = entropy.mean(dim=dims_to_reduce)
    
    return entropy

# ===== MODEL LOADING =====
print("\n📚 Loading model and tokenizer...")
MODEL_NAME = "distilgpt2"  # Small model for faster testing

try:
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Successfully loaded {MODEL_NAME}")

    # Create sample input
    print("\n📝 Creating sample inputs...")
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Run model to get attention patterns
    print("\n🧠 Executing forward pass to get attention patterns...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

    # Extract attention tensors
    attention_tensors = outputs.attentions
    print(f"✅ Got {len(attention_tensors)} attention layers")
    
    # ===== ENTROPY CALCULATION =====
    print("\n🧮 Calculating head entropy...")
    entropy_values = {}
    for layer_idx, layer_attention in enumerate(attention_tensors):
        # Use our calculate_head_entropy function
        layer_entropy = calculate_head_entropy(layer_attention)
        entropy_values[layer_idx] = layer_entropy
        print(f"   Layer {layer_idx}: {layer_entropy.shape}, min={layer_entropy.min().item():.4f}, max={layer_entropy.max().item():.4f}")

    # ===== VISUALIZATION =====
    print("\n📊 Creating entropy visualization...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create entropy heatmap data
    entropy_data = torch.stack([entropy_values[layer] for layer in range(len(entropy_values))]).cpu().numpy()
    
    # Plot heatmap
    im = ax.imshow(entropy_data, cmap="viridis")
    ax.set_title("Attention Entropy Heatmap")
    
    # Set proper colormap limits
    im.set_clim(0.0, np.max(entropy_data))
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Entropy')
    
    # Add labels
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer Index')
    
    # Save the visualization
    os.makedirs("test_output", exist_ok=True)
    plt.savefig("test_output/entropy_heatmap.png")
    plt.close()
    print("✅ Saved entropy heatmap to test_output/entropy_heatmap.png")
    
    # ===== PRUNING MASK GENERATION =====
    print("\n✂️ Generating pruning mask...")
    
    # Create dummy gradient values
    num_layers = len(entropy_values)
    num_heads = entropy_values[0].shape[0]
    grad_norm_values = torch.ones((num_layers, num_heads), device='cpu')
    
    # Create pruning mask based on entropy (higher entropy = more likely to prune)
    flat_entropy = torch.cat([entropy_values[layer].view(-1) for layer in range(num_layers)])
    
    # Set pruning threshold at 70th percentile of entropy
    prune_threshold = torch.quantile(flat_entropy, 0.7).item()  # Convert to scalar
    pruning_mask = torch.zeros((num_layers, num_heads), dtype=torch.bool)
    
    for layer in range(num_layers):
        # For each head in the layer, get the mean entropy across positions
        head_entropies = entropy_values[layer].mean(dim=0) if entropy_values[layer].dim() > 1 else entropy_values[layer]
        
        # Compare against threshold
        for head in range(len(head_entropies)):
            if head_entropies[head].item() > prune_threshold:
                pruning_mask[layer, head] = True
    
    pruned_count = pruning_mask.sum().item()
    print(f"✅ Generated pruning mask with {pruned_count} out of {num_layers * num_heads} heads marked for pruning")
    
    # ===== APPLY PRUNING MASK =====
    print("\n🧬 Applying pruning mask to model...")
    
    # Apply pruning mask to model weights
    with torch.no_grad():
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                if pruning_mask[layer_idx, head_idx]:
                    # Find attention layer
                    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                        # GPT-2 style: get the layer and zero out the weights for this head
                        layer = model.transformer.h[layer_idx]
                        if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_proj'):
                            # Zero out the output projection weights for this head
                            head_size = layer.attn.c_proj.weight.size(0) // model.config.n_head
                            start_idx = head_idx * head_size
                            end_idx = (head_idx + 1) * head_size
                            
                            # Zero out weights
                            layer.attn.c_proj.weight[start_idx:end_idx, :] = 0.0
                            if hasattr(layer.attn.c_proj, 'bias') and layer.attn.c_proj.bias is not None:
                                layer.attn.c_proj.bias[start_idx:end_idx] = 0.0
    
    print(f"✅ Applied pruning to {pruning_mask.sum().item()} attention heads")
    
    # ===== FINE-TUNING =====
    print("\n🔄 Simulating fine-tuning of pruned model...")
    
    # Configure a small fine-tuning run (just a few steps for validation)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Create a minimal dataset for fine-tuning simulation
    fine_tuning_prompts = [
        "The transformer model",
        "Neural plasticity is",
        "AI systems can adapt",
        "GPT models use attention"
    ]
    
    # Run a few fine-tuning steps
    for step in range(3):  # Just 3 steps for demonstration
        # Select a random prompt
        prompt = fine_tuning_prompts[step % len(fine_tuning_prompts)]
        
        # Tokenize and prepare inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Set inputs as targets for causal LM
        inputs["labels"] = inputs["input_ids"].clone()
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update model weights
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"   Step {step+1}, Loss: {loss.item():.4f}")
    
    # Set model back to evaluation mode
    model.eval()
    print("✅ Fine-tuning simulation completed")
    
    # ===== TEST GENERATION =====
    print("\n📝 Testing text generation with pruned model...")
    
    generation_prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In a world where AI"
    ]
    
    for prompt in generation_prompts:
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=50,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
    
    # ===== ATTENTION VISUALIZATION =====
    print("\n👁️ Visualizing attention patterns after pruning...")
    
    # Get attention patterns after pruning
    with torch.no_grad():
        pruned_outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), output_attentions=True)
        
    pruned_attentions = pruned_outputs.attentions
    
    # Create a simple visualization of first layer attention (first head)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract first layer, first head attention
    attn_to_viz = pruned_attentions[0][0, 0].cpu().numpy()
    
    # Plot as heatmap
    im = ax.imshow(attn_to_viz, cmap='viridis')
    ax.set_title("Attention Pattern (Layer 0, Head 0)")
    plt.colorbar(im, label='Attention Weight')
    
    # Set axis labels
    ax.set_xlabel('Target Token Position')
    ax.set_ylabel('Source Token Position')
    
    # Save visualization
    plt.savefig("test_output/attention_pattern.png")
    plt.close()
    print("✅ Saved attention pattern visualization to test_output/attention_pattern.png")
    
    print("\n💯 All Neural Plasticity components executed successfully!")
    
except Exception as e:
    print(f"\n❌ Error during execution: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===== FINAL TIMING =====
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️ Total execution time: {elapsed_time:.2f} seconds")