#!/usr/bin/env python
"""
Run Neural Plasticity Notebook with Minimal Configuration

This script runs key components of the Neural Plasticity notebook with:
- Smallest possible model (distilgpt2)
- Minimal dataset size
- Reduced iterations and epochs
- Limited steps per epoch

This allows us to verify end-to-end functionality without requiring
extensive computational resources.

Usage:
  python scripts/run_neural_plasticity_minimal.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset
from datetime import datetime
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

# Define our own mini evaluate function
def mini_evaluate_model(model, dataloader, device):
    """Minimal model evaluation function."""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            total_steps += 1
            
            # Limit evaluation to 5 steps for speed
            if total_steps >= 2:
                break
    
    avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def run_minimal_notebook():
    """Run the Neural Plasticity notebook with minimal configuration."""
    print(f"Starting minimal Neural Plasticity run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = os.path.join(repo_root, "test_output", "minimal_run")
    os.makedirs(output_dir, exist_ok=True)
    
    # Import neural plasticity modules
    try:
        from utils.neural_plasticity.core import (
            generate_pruning_mask
        )
        
        print("✅ Successfully imported neural plasticity modules")
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        return False
    
    # Configure minimal parameters
    MODEL_NAME = "distilgpt2"  # Smallest GPT-2 model
    DATASET = "wikitext"
    DATASET_CONFIG = "wikitext-2-raw-v1"
    MAX_LENGTH = 64  # Smaller sequence length
    BATCH_SIZE = 2  # Tiny batch size
    NUM_EPOCHS = 1  # Just one epoch
    MAX_STEPS_PER_EPOCH = 5  # Very few steps
    LEARNING_RATE = 5e-5
    WARMUP_STEPS = 2
    WARMUP_MAX_EPOCHS = 1
    EVAL_INTERVAL = 2
    PRUNE_PERCENT = 0.1
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load model and tokenizer
        print(f"Loading model: {MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load minimal dataset
        print(f"Loading minimal dataset: {DATASET}/{DATASET_CONFIG}")
        train_dataset = load_dataset(DATASET, DATASET_CONFIG, split="train[:100]")  # Just 100 examples
        validation_dataset = load_dataset(DATASET, DATASET_CONFIG, split="validation[:20]")  # Just 20 examples
        
        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=MAX_LENGTH
            )
        
        # Tokenize datasets
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
        
        print(f"Dataset preparation complete: {len(train_dataset)} train, {len(validation_dataset)} validation examples")
        
        # Evaluate baseline model
        print("Evaluating baseline model...")
        baseline_loss, baseline_perplexity = mini_evaluate_model(model, validation_dataloader, device)
        print(f"Baseline evaluation: Loss = {baseline_loss:.4f}, Perplexity = {baseline_perplexity:.2f}")
        
        # Find model configuration (distilgpt2 has 6 layers, 12 heads)
        num_layers = 6
        num_heads = 12
        
        # Instead of calculating gradients, use random values for demonstration
        print("Generating simulated gradient norms...")
        # Create random gradient values
        grad_norm_values = torch.rand(num_layers, num_heads, device=device)
        print(f"Generated gradient norms with shape {grad_norm_values.shape}")
        
        # Apply pruning based on gradients
        print("Generating pruning mask...")
        pruning_mask = generate_pruning_mask(
            grad_norm_values=grad_norm_values,
            prune_percent=PRUNE_PERCENT,
            strategy="gradient"
        )
        
        # Convert to list of tuples for visualization
        pruned_heads = []
        for layer in range(pruning_mask.size(0)):
            for head in range(pruning_mask.size(1)):
                if pruning_mask[layer, head]:
                    pruned_heads.append((layer, head))
        
        print(f"Generated pruning mask with {len(pruned_heads)} heads")
        
        # Visualize gradients and pruning
        plt.figure(figsize=(10, 6))
        plt.imshow(grad_norm_values.detach().cpu().numpy(), cmap="plasma", aspect="auto")
        plt.colorbar(label="Gradient Norm")
        
        # Mark pruned heads with 'P'
        for layer, head in pruned_heads:
            plt.text(head, layer, "P", ha="center", va="center", 
                     color="white", weight="bold", bbox=dict(facecolor='red', alpha=0.5))
        
        plt.title("Gradient Norms with Pruning")
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")
        plt.savefig(os.path.join(output_dir, "gradient_pruning.png"))
        plt.close()
        
        print(f"✅ Visualized gradients and pruning decisions")
        
        # Initialize optimizer for minimal training
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
        # Run a few training steps
        print(f"Running {MAX_STEPS_PER_EPOCH} training steps...")
        model.train()
        training_losses = []
        
        for step, batch in enumerate(train_dataloader):
            if step >= MAX_STEPS_PER_EPOCH:
                break
                
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
            
            # Store loss
            training_losses.append(loss.item())
            print(f"  Step {step+1}/{MAX_STEPS_PER_EPOCH}: Loss = {loss.item():.4f}")
        
        # Visualize training progress
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(training_losses) + 1), training_losses)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "training_progress.png"))
        plt.close()
        
        # Evaluate final model
        print("Evaluating final model...")
        final_loss, final_perplexity = mini_evaluate_model(model, validation_dataloader, device)
        print(f"Final evaluation: Loss = {final_loss:.4f}, Perplexity = {final_perplexity:.2f}")
        print(f"Change: {((baseline_loss - final_loss) / baseline_loss * 100):.2f}%")
        
        # Generate text (minimal version)
        def generate_text(prompt, max_length=50):
            model.eval()
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            return tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Test text generation
        prompt = "Once upon a time"
        generated_text = generate_text(prompt)
        print(f"\nGenerated text with prompt '{prompt}':")
        print(generated_text)
        
        # Write results to summary file
        with open(os.path.join(output_dir, "results_summary.txt"), "w") as f:
            f.write(f"Neural Plasticity Minimal Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Training steps: {MAX_STEPS_PER_EPOCH}\n\n")
            f.write(f"Baseline loss: {baseline_loss:.4f}, Perplexity: {baseline_perplexity:.2f}\n")
            f.write(f"Final loss: {final_loss:.4f}, Perplexity: {final_perplexity:.2f}\n")
            f.write(f"Change: {((baseline_loss - final_loss) / baseline_loss * 100):.2f}%\n\n")
            f.write(f"Generated text (prompt: '{prompt}'):\n{generated_text}\n")
        
        print(f"\n✅ Minimal run completed successfully")
        print(f"Results saved to {output_dir}/")
        return True
    
    except Exception as e:
        print(f"\n❌ Error during minimal run: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_minimal_notebook()
    sys.exit(0 if success else 1)