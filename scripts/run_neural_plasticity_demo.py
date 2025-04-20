#!/usr/bin/env python
"""
Run Neural Plasticity Demo - Minimal Notebook Implementation

This script runs the same functionality as the NeuralPlasticityDemo notebook,
but with minimal data to verify everything works properly:
- Load model and datasets
- Calculate entropy and track gradients
- Generate pruning masks
- Apply pruning to model
- Fine-tune the pruned model
- Evaluate perplexity before and after

Usage:
  source .venv/bin/activate
  python scripts/run_neural_plasticity_demo.py
"""

import os
import sys
import platform
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    default_data_collator,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Create output directory
os.makedirs("test_output", exist_ok=True)

print("="*50)
print("NEURAL PLASTICITY DEMO")
print("="*50)

# ===== ENVIRONMENT DETECTION =====
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"

if IS_APPLE_SILICON:
    print("🍎 Apple Silicon detected - applying optimizations")
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
    except (AttributeError, RuntimeError) as e:
        print(f"⚠️ Could not set all PyTorch safeguards: {e}")
    
    device = torch.device("cpu")
    print("💻 Using CPU (forced on Apple Silicon)")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

# ===== CONFIGURATION =====
MODEL_NAME = "distilgpt2"
BATCH_SIZE = 4
MAX_LENGTH = 128
LEARNING_RATE = 5e-5
PRUNE_PERCENT = 0.1
ENTROPY_PERCENTILE = 70
GRADIENT_PERCENTILE = 30

# ===== SKIPPING DATASETS IMPORT =====
print("\nSkipping datasets import, will use synthetic data only...")

# ===== FUNCTIONS =====
def calculate_head_entropy(attention_maps, eps=1e-6):
    """Calculate entropy of attention patterns."""
    # Handle tensor properly
    if attention_maps.is_cuda and device.type == "cuda":
        pass  # Keep on GPU
    elif IS_APPLE_SILICON:
        attention_maps = attention_maps.detach().cpu()  # Force to CPU
    
    # Ensure numerical stability
    attn_probs = torch.where(
        torch.isfinite(attention_maps),
        attention_maps,
        torch.ones_like(attention_maps) * eps
    )
    attn_probs = attn_probs.clamp(min=eps)
    
    # Normalize
    attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)
    
    # Average over batch dimensions
    dims_to_reduce = list(range(entropy.dim() - 2))
    if dims_to_reduce:
        entropy = entropy.mean(dim=dims_to_reduce)
    
    return entropy

def calculate_head_gradients(model, dataloader, num_batches=2):
    """Calculate gradient norms for each head."""
    # Get model structure
    if hasattr(model, 'config'):
        num_layers = model.config.n_layer
        num_heads = model.config.n_head
    else:
        # Fallback for different architectures
        num_layers = len(model.transformer.h)
        num_heads = model.transformer.h[0].attn.n_head
    
    grad_norms = torch.zeros((num_layers, num_heads), device=device)
    
    # Track gradients
    model.train()
    batch_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        # Process batch
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        # Extract gradients from attention output projections
        for layer_idx in range(num_layers):
            layer = model.transformer.h[layer_idx]
            if hasattr(layer.attn, 'c_proj') and layer.attn.c_proj.weight.grad is not None:
                head_size = layer.attn.c_proj.weight.size(0) // num_heads
                
                # Extract gradient for each head
                for head_idx in range(num_heads):
                    start_idx = head_idx * head_size
                    end_idx = (head_idx + 1) * head_size
                    head_grad = layer.attn.c_proj.weight.grad[start_idx:end_idx]
                    grad_norms[layer_idx, head_idx] += head_grad.norm().item()
        
        model.zero_grad()
        batch_count += 1
    
    # Average gradients
    if batch_count > 0:
        grad_norms /= batch_count
    
    return grad_norms

def generate_pruning_mask(entropy_values, grad_norm_values):
    """Generate pruning mask based on entropy and gradient values."""
    # Create pruning mask
    num_layers, num_heads = grad_norm_values.shape
    pruning_mask = torch.zeros((num_layers, num_heads), dtype=torch.bool)
    
    # Flatten tensors for percentile calculation
    flat_entropy = torch.cat([entropy_values[layer] for layer in range(num_layers)])
    flat_gradients = grad_norm_values.view(-1)
    
    # Calculate thresholds
    entropy_threshold = torch.quantile(flat_entropy, ENTROPY_PERCENTILE/100).item()
    gradient_threshold = torch.quantile(flat_gradients, GRADIENT_PERCENTILE/100).item()
    
    # Generate combined scores for each head
    scores = torch.zeros((num_layers, num_heads))
    for layer in range(num_layers):
        layer_entropy = entropy_values[layer]
        for head in range(num_heads):
            # Normalized entropy (higher is worse)
            norm_entropy = (layer_entropy[head] - flat_entropy.min()) / (flat_entropy.max() - flat_entropy.min() + 1e-8)
            # Normalized gradient (higher is better, so invert)
            norm_grad = 1.0 - (grad_norm_values[layer, head] - flat_gradients.min()) / (flat_gradients.max() - flat_gradients.min() + 1e-8)
            # Combined score (higher = more likely to prune)
            scores[layer, head] = norm_entropy.item() * 0.6 + norm_grad.item() * 0.4
    
    # Select top heads to prune
    total_heads = num_layers * num_heads
    target_prune_count = int(total_heads * PRUNE_PERCENT)
    flat_scores = scores.view(-1)
    _, indices = torch.topk(flat_scores, k=target_prune_count)
    
    # Mark heads for pruning
    for idx in indices:
        layer = idx.item() // num_heads
        head = idx.item() % num_heads
        pruning_mask[layer, head] = True
    
    return pruning_mask

def apply_pruning_mask(model, pruning_mask):
    """Apply pruning mask to model weights."""
    num_layers, num_heads = pruning_mask.shape
    pruned_heads = []
    
    with torch.no_grad():
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                if pruning_mask[layer_idx, head_idx]:
                    # Add to pruned list
                    pruned_heads.append((layer_idx, head_idx))
                    
                    # Zero out weights for this head
                    layer = model.transformer.h[layer_idx]
                    if hasattr(layer.attn, 'c_proj'):
                        # Get dimensions for this head
                        head_size = layer.attn.c_proj.weight.size(0) // num_heads
                        start_idx = head_idx * head_size
                        end_idx = (head_idx + 1) * head_size
                        
                        # Zero out weights and bias
                        layer.attn.c_proj.weight[start_idx:end_idx, :] = 0.0
                        if hasattr(layer.attn.c_proj, 'bias') and layer.attn.c_proj.bias is not None:
                            layer.attn.c_proj.bias[start_idx:end_idx] = 0.0
    
    return pruned_heads

def evaluate_model(model, dataloader, max_steps=5):
    """Evaluate model performance."""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if total_steps >= max_steps:
                break
                
            # Process batch
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            total_steps += 1
    
    avg_loss = total_loss / total_steps if total_steps > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }

def visualize_entropy(entropy_values, title):
    """Visualize entropy heatmap."""
    plt.figure(figsize=(10, 6))
    
    # Create entropy heatmap data
    num_layers = len(entropy_values)
    entropy_data = torch.stack([entropy_values[layer] for layer in range(num_layers)]).cpu().numpy()
    
    # Plot heatmap
    plt.imshow(entropy_data, cmap="viridis")
    plt.colorbar(label='Entropy')
    plt.title(title)
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    
    plt.savefig("test_output/entropy_heatmap.png")
    plt.close()
    
def visualize_pruning_mask(pruning_mask, title):
    """Visualize pruning mask."""
    plt.figure(figsize=(10, 6))
    
    # Plot mask
    plt.imshow(pruning_mask.cpu().numpy(), cmap="binary")
    plt.colorbar(label='Pruned')
    plt.title(title)
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    
    plt.savefig("test_output/pruning_mask.png")
    plt.close()

# ===== LOAD MODEL AND TOKENIZER =====
print("\nLoading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set pad token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {MODEL_NAME}")

# ===== LOAD DATASET =====
print("\nLoading dataset...")
try:
    DATASET_CONFIG = "wikitext-2-raw-v1"
    train_dataset = load_dataset("wikitext", DATASET_CONFIG, split="train")
    validation_dataset = load_dataset("wikitext", DATASET_CONFIG, split="validation")
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LENGTH
        )
    
    # Process datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Add labels for language modeling
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    train_dataset = train_dataset.map(add_labels)
    validation_dataset = validation_dataset.map(add_labels)
    
    # Set format
    train_dataset = train_dataset.with_format("torch")
    validation_dataset = validation_dataset.with_format("torch")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=default_data_collator
    )
    
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=default_data_collator
    )
    
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(validation_dataset)} examples")
    dataset_loaded = True

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Using minimal synthetic dataset instead")
    
    # Create minimal synthetic dataset
    import random
    
    # Generate random tokens
    def generate_random_tokens(vocab_size=50000, length=MAX_LENGTH):
        return torch.randint(100, vocab_size-100, (length,))
    
    # Create synthetic dataset
    synthetic_samples = 100
    input_ids_list = [generate_random_tokens() for _ in range(synthetic_samples)]
    
    # Create dataloader
    def collate_fn(batch):
        # Pad to max length
        padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
        return {
            "input_ids": padded,
            "attention_mask": (padded != tokenizer.pad_token_id).long(),
            "labels": padded.clone()
        }
    
    train_dataloader = DataLoader(
        input_ids_list[:80], 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    validation_dataloader = DataLoader(
        input_ids_list[80:], 
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn
    )
    
    print(f"Created synthetic dataset: {synthetic_samples} examples")
    dataset_loaded = True

# ===== EVALUATE BASELINE MODEL =====
print("\nEvaluating baseline model...")
baseline_results = evaluate_model(model, validation_dataloader)
baseline_loss = baseline_results["loss"]
baseline_perplexity = baseline_results["perplexity"]
print(f"Baseline Loss: {baseline_loss:.4f}, Perplexity: {baseline_perplexity:.2f}")

# ===== CALCULATE ENTROPY =====
print("\nCalculating attention entropy...")
# Get a batch for attention analysis
batch = next(iter(validation_dataloader))
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None

# Get attention patterns
model.eval()
with torch.no_grad():
    if attention_mask is not None:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    else:
        outputs = model(input_ids=input_ids, output_attentions=True)

# Extract attention tensors
attention_tensors = outputs.attentions

# Calculate entropy for each head
entropy_values = {}
for layer_idx, layer_attention in enumerate(attention_tensors):
    layer_entropy = calculate_head_entropy(layer_attention)
    entropy_values[layer_idx] = layer_entropy
    print(f"Layer {layer_idx} entropy - min: {layer_entropy.min().item():.4f}, max: {layer_entropy.max().item():.4f}")

# Visualize entropy
visualize_entropy(entropy_values, "Attention Entropy Heatmap")
print("Entropy visualization saved to test_output/entropy_heatmap.png")

# ===== CALCULATE GRADIENTS =====
print("\nCalculating head gradients...")
grad_norm_values = calculate_head_gradients(model, train_dataloader, num_batches=2)
print(f"Gradient shape: {grad_norm_values.shape}")
print(f"Gradient range: {grad_norm_values.min().item():.6f} to {grad_norm_values.max().item():.6f}")

# ===== GENERATE PRUNING MASK =====
print("\nGenerating pruning mask...")
pruning_mask = generate_pruning_mask(entropy_values, grad_norm_values)
pruned_count = pruning_mask.sum().item()
total_heads = pruning_mask.numel()
print(f"Pruning {pruned_count} out of {total_heads} heads ({100*pruned_count/total_heads:.1f}%)")

# Visualize pruning mask
visualize_pruning_mask(pruning_mask, "Pruning Mask")
print("Pruning mask visualization saved to test_output/pruning_mask.png")

# ===== APPLY PRUNING =====
print("\nApplying pruning to model...")
pruned_heads = apply_pruning_mask(model, pruning_mask)
print(f"Pruned {len(pruned_heads)} heads")

# ===== EVALUATE AFTER PRUNING =====
print("\nEvaluating model after pruning...")
pruned_results = evaluate_model(model, validation_dataloader)
pruned_loss = pruned_results["loss"]
pruned_perplexity = pruned_results["perplexity"]
print(f"After pruning - Loss: {pruned_loss:.4f}, Perplexity: {pruned_perplexity:.2f}")
print(f"Change from baseline: {((pruned_loss - baseline_loss) / baseline_loss * 100):+.2f}%")

# ===== FINE-TUNE PRUNED MODEL =====
print("\nFine-tuning pruned model...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=10, 
    num_training_steps=100
)

# Train for a few steps
training_steps = min(100, len(train_dataloader))
for step, batch in enumerate(train_dataloader):
    if step >= training_steps:
        break
        
    # Process batch
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    outputs = model(**batch)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    # Print progress
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# ===== EVALUATE AFTER FINE-TUNING =====
print("\nEvaluating fine-tuned model...")
model.eval()
fine_tuned_results = evaluate_model(model, validation_dataloader)
fine_tuned_loss = fine_tuned_results["loss"]
fine_tuned_perplexity = fine_tuned_results["perplexity"]
print(f"After fine-tuning - Loss: {fine_tuned_loss:.4f}, Perplexity: {fine_tuned_perplexity:.2f}")
print(f"Change from baseline: {((fine_tuned_loss - baseline_loss) / baseline_loss * 100):+.2f}%")
print(f"Recovery from pruning: {((pruned_loss - fine_tuned_loss) / pruned_loss * 100):+.2f}%")

# ===== GENERATE TEXT =====
print("\nGenerating text with pruned and fine-tuned model...")
prompts = [
    "Once upon a time",
    "The meaning of life is"
]

for prompt in prompts:
    # Encode prompt
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
    
    # Decode text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")

# ===== SUMMARY =====
print("\n===== NEURAL PLASTICITY SUMMARY =====")
print(f"Initial perplexity: {baseline_perplexity:.2f}")
print(f"After pruning: {pruned_perplexity:.2f} ({((pruned_perplexity - baseline_perplexity) / baseline_perplexity * 100):+.2f}%)")
print(f"After fine-tuning: {fine_tuned_perplexity:.2f} ({((fine_tuned_perplexity - baseline_perplexity) / baseline_perplexity * 100):+.2f}%)")
print(f"Heads pruned: {pruned_count} out of {total_heads} ({100*pruned_count/total_heads:.1f}%)")

print("\n✅ Neural Plasticity demo completed successfully!")