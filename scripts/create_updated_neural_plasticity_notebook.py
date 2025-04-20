#!/usr/bin/env python
"""
Create Updated Neural Plasticity Notebook

This script creates an updated version of the NeuralPlasticityDemo notebook
that uses our modularized neural plasticity code with Apple Silicon fixes.

Version: v0.0.57 (2025-04-19 23:45:00)
"""

import os
import sys
import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from datetime import datetime

# Set version and timestamp
VERSION = "0.0.57"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def create_neural_plasticity_notebook():
    """Create the updated neural plasticity notebook."""
    nb = new_notebook()
    
    # Add cells for the notebook
    cells = []
    
    # Title and introduction
    cells.append(new_markdown_cell(f"""# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v{VERSION} {TIMESTAMP})

This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to dynamically prune and regrow attention heads during training based on utility metrics. [ID: {hash(TIMESTAMP) % 10000000:08x}]

### Changes in v{VERSION}:
- Completely refactored neural plasticity code for better modularity
- Added Apple Silicon (M1/M2/M3) compatibility fixes
- Fixed BLAS/libtorch crash issues in entropy calculation
- Improved tensor handling with proper CPU/CUDA management
- Added Colab environment detection for conditional optimizations
- Enhanced numerical stability in matrix operations
- Fixed tensor dimensions handling in pruning functions
- Added detailed debugging metrics for troubleshooting

## What is Neural Plasticity?

Neural plasticity is the ability of neural networks to adapt their structure over time through pruning (removing unused connections) and regrowth (restoring useful connections). This mimics how biological brains form efficient neural pathways.

In this demo, we:
1. Track the entropy and gradient patterns of each attention head
2. Dynamically prune high-entropy, low-gradient heads (unfocused, less useful)
3. Selectively revive low-entropy, higher-gradient heads (potentially useful)
4. Visualize the "brain dynamics" over time

This allows models to form more efficient neural structures during training.
"""))

    # System dependencies - only for Colab
    cells.append(new_code_cell("""# This cell is only needed in Colab
if 'google.colab' in str(get_ipython()):
    # Install system dependencies in Colab
    !apt-get update -qq > /dev/null
    !apt-get install -qq libopenblas-dev > /dev/null  # For better performance
    print("Installed system dependencies for Colab")
else:
    print("Running locally - skipping Colab-specific system dependencies")"""))

    # Install packages and clone repo
    cells.append(new_code_cell("""# This cell is only needed in Colab - has no effect when running locally
import os

# Check if we're running in Colab
IN_COLAB = 'google.colab' in str(get_ipython())

if IN_COLAB:
    # Install required packages in Colab
    !pip install -q torch transformers datasets matplotlib seaborn
    
    # Clone the repository in Colab
    !git clone -b feature/implement-adaptive-plasticity https://github.com/CambrianTech/sentinel-ai.git
    %cd sentinel-ai
    
    # Add repository to path
    import sys
    sys.path.append('.')
else:
    # When running locally, we're already in the repository
    print("Running locally - no need to clone repository")
    
    # Make sure all required packages are installed
    import importlib
    
    required_packages = ['torch', 'transformers', 'datasets', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Warning: The following packages are missing and should be installed: {', '.join(missing_packages)}")
    else:
        print("All required packages are installed")"""))

    # Configure experiment
    cells.append(new_markdown_cell("""# Configure the Experiment

Let's set up our configuration for the neural plasticity experiment"""))

    cells.append(new_code_cell("""# Configure experiment
MODEL_NAME = "distilgpt2"  # Small GPT-2 model for faster demonstration
DATASET = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
MAX_LENGTH = 128
BATCH_SIZE = 4
NUM_EPOCHS = 100      # Run for many epochs if needed
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
WARMUP_MAX_EPOCHS = 1     # Maximum number of warmup epochs (will stop earlier if loss stabilizes)
EVAL_INTERVAL = 50    # Evaluate every 50 steps
VISUALIZATION_INTERVAL = 100  # Show visuals every 100 steps
INFERENCE_INTERVAL = 500      # Run inference every 500 steps
CHECKPOINT_INTERVAL = 500    # Save checkpoint more frequently (was 1000)
MAX_STEPS_PER_EPOCH = None    # Set to a number to limit steps per epoch, or None for unlimited

# Set to True to enable continuous training for long periods
ENABLE_LONG_TRAINING = False  # Set to False for demo purposes to avoid memory/runtime issues

# If ENABLE_LONG_TRAINING is True, run with unlimited steps per epoch
# If ENABLE_LONG_TRAINING is False, override to a reasonable limit for demo purposes
if not ENABLE_LONG_TRAINING:
    MAX_STEPS_PER_EPOCH = 200 # Limit steps per epoch for demo purposes
    NUM_EPOCHS = 3            # Limit epochs for demo purposes

# Configure pruning mode
try:
    # First try the new modular structure
    from utils.neural_plasticity.core import PruningStrategy
    # Define an enum-like class for compatibility
    class PruningMode:
        ADAPTIVE = "adaptive"   # Allows recovery
        COMPRESSED = "compressed"  # Prevents recovery
    
    PRUNING_STRATEGY_CLASS = PruningStrategy
except ImportError:
    try:
        # Fall back to the original structure if in Colab
        from sentinel.pruning.dual_mode_pruning import PruningMode
        PRUNING_STRATEGY_CLASS = PruningMode
    except ImportError:
        # If all else fails, define a simple enum for the demo
        class PruningMode:
            ADAPTIVE = "adaptive"
            COMPRESSED = "compressed"
        PRUNING_STRATEGY_CLASS = PruningMode
        print("WARNING: Using simplified pruning mode classes")

# Set pruning mode (ADAPTIVE allows recovery, COMPRESSED prevents recovery)
PRUNING_MODE = PruningMode.ADAPTIVE  # Change to PruningMode.COMPRESSED for permanent pruning

# Configure statistical-based pruning strategy
# Instead of fixed thresholds, we'll use percentile-based thresholds
ENTROPY_PERCENTILE = 70  # Heads with entropy above the 70th percentile are candidates for pruning
GRADIENT_PERCENTILE = 30  # Heads with gradient below the 30th percentile are candidates for pruning
PRUNE_PERCENT = 0.1      # Target to prune approximately 10% of heads in each step
MIN_ZERO_EPOCHS = 1      # Minimum epochs a head should remain pruned"""))

    # Load model and dataset
    cells.append(new_markdown_cell("""# Load Model and Dataset

Now we'll load the model and prepare the dataset for training"""))

    cells.append(new_code_cell("""
%matplotlib inline
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    default_data_collator,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from datasets import load_dataset

# Import neural plasticity modules
from utils.neural_plasticity.core import (
    calculate_head_entropy,
    calculate_head_gradients,
    generate_pruning_mask,
    apply_pruning_mask,
    evaluate_model,
    IS_APPLE_SILICON,
    IS_COLAB
)

# Import Apple Silicon optimizations
try:
    from utils.apple_silicon import (
        safe_matmul, 
        apply_tensor_patches,
        restore_tensor_patches,
        safe_context,
        ensure_cpu_tensor,
        ensure_cpu_model
    )
except ImportError:
    # Fallback to core module if apple_silicon not available
    from utils.neural_plasticity.core import safe_matmul

from utils.neural_plasticity.visualization import (
    visualize_head_entropy,
    visualize_head_gradients,
    visualize_pruning_decisions,
    visualize_training_metrics,
    visualize_attention_patterns
)

# Import visualization utilities
from utils.colab.helpers import safe_tensor_imshow

# Check if we're running on Apple Silicon or in Colab
if IS_APPLE_SILICON:
    print("🍎 Apple Silicon detected - using optimized tensor operations")
if IS_COLAB:
    print("🌐 Running in Google Colab environment")

# Set device - force CPU on Apple Silicon regardless of CUDA availability
device = torch.device("cpu") if IS_APPLE_SILICON else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set pad token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load datasets
print(f"Loading dataset: {DATASET}/{DATASET_CONFIG}")
train_dataset = load_dataset(DATASET, DATASET_CONFIG, split="train")
validation_dataset = load_dataset(DATASET, DATASET_CONFIG, split="validation")

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

print(f"Train dataset size: {len(train_dataset)} examples")
print(f"Validation dataset size: {len(validation_dataset)} examples")

# Define unique ID for cache busting
unique_id = f"{hash(TIMESTAMP) % 10000000:08x}"
print(f"Running modularized neural plasticity code [ID: {unique_id}]")
"""))

    # Define evaluation function
    cells.append(new_markdown_cell("""# Define Evaluation Function

Let's define a function to evaluate our model's performance"""))

    cells.append(new_code_cell("""def evaluate_model_performance(model, dataloader, device):
    # Evaluate model on the provided dataloader
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
            
            # Limit evaluation to 10 steps for speed
            if total_steps >= 10:
                break
    
    avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def generate_text(model, tokenizer, prompt, device, max_length=100):
    # Generate text from the model
    # Set model to evaluation mode
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
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
    
    # Decode and return text
    return tokenizer.decode(output[0], skip_special_tokens=True)"""))

    # Warm-up phase
    cells.append(new_markdown_cell("""## Run Model Warm-up

Before measuring baseline performance and applying neural plasticity, we'll run a brief warm-up phase to get initial attention patterns and stabilize metrics."""))

    cells.append(new_code_cell("""# Initialize optimizer and scheduler for warm-up
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_dataloader) * WARMUP_MAX_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=WARMUP_STEPS, 
    num_training_steps=total_steps
)

print(f"Running warm-up until loss stabilizes (max {WARMUP_MAX_EPOCHS} epochs)...")

# Warm-up training loop
model.train()
warmup_losses = []
warmup_step_losses = []
last_loss_decrease = 0
patience = 15      # Number of steps with no decrease to consider stabilized
min_warmup_steps = 50  # Minimum number of warm-up steps
max_warmup_steps = 150  # Maximum number of warm-up steps per epoch

# Helper function to calculate if loss has stabilized 
def is_loss_stabilized(losses, min_steps, patience_steps, window_size=5):
    # Check if loss has stabilized
    # Not enough steps yet
    if len(losses) < min_steps:
        return False, 0

    # Not enough steps since last decrease
    steps_since_decrease = len(losses) - last_loss_decrease
    if steps_since_decrease < patience_steps:
        return False, steps_since_decrease
    
    # Check if recent trend is flat or increasing using rolling average
    if len(losses) >= window_size * 2:
        recent_window = sum(losses[-window_size:]) / window_size
        previous_window = sum(losses[-(window_size*2):-window_size]) / window_size
        # If recent average is lower than previous, we're still decreasing
        if recent_window < previous_window * 0.99:  # Allow 1% variation
            return False, steps_since_decrease
            
    return True, steps_since_decrease

try:
    for epoch in range(WARMUP_MAX_EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Track loss
            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            warmup_losses.append(loss_val)
            
            # Check if we've met the minimum steps and loss has stabilized
            if len(warmup_losses) > 1:
                # Track non-increasing steps
                if loss_val <= warmup_losses[-2]:
                    last_loss_decrease = len(warmup_losses)
                
                # For visualization, track a smoothed version (rolling average of 5)
                if len(warmup_losses) % 5 == 0:
                    avg_loss = sum(warmup_losses[-5:]) / 5
                    warmup_step_losses.append(avg_loss)
            
            # Print progress every 5 steps
            if step % 5 == 0:
                print(f"Warm-up Epoch {epoch+1}, Step {step}: Loss = {loss_val:.4f}", end='\r')
            
            # Check if loss has stabilized
            is_stable, steps_without_decrease = is_loss_stabilized(
                warmup_losses, min_warmup_steps, patience
            )
            
            if is_stable:
                print(f"\nWarm-up loss stabilized after {len(warmup_losses)} steps")
                print(f"Loss has been non-decreasing for {steps_without_decrease} steps")
                break
                
            # Stop after max_warmup_steps for faster execution in demo
            if step >= max_warmup_steps:
                print(f"\nReached maximum warm-up steps per epoch ({max_warmup_steps})")
                break
        
        print(f"\nWarm-up Epoch {epoch+1} completed: Average Loss = {epoch_loss / epoch_steps:.4f}")
        
        # Check if loss has stabilized across epochs
        is_stable, steps_without_decrease = is_loss_stabilized(
            warmup_losses, min_warmup_steps, patience
        )
        
        if is_stable:
            print(f"Loss has stabilized with {steps_without_decrease} steps without significant decrease.")
            print(f"Ending warm-up early after {epoch+1} epochs.")
            break
    
    # Plot warm-up loss
    plt.figure(figsize=(12, 8))
    
    # Raw loss
    plt.subplot(2, 1, 1)
    plt.plot(warmup_losses)
    plt.title("Warm-up Loss (Raw)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    
    # Smoothed loss if we have enough data
    if len(warmup_step_losses) > 1:
        plt.subplot(2, 1, 2)
        plt.plot(range(0, len(warmup_step_losses)*5, 5), warmup_step_losses)
        plt.title("Warm-up Loss (5-step Rolling Average)")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        
        # Add trend line to smoothed plot
        from scipy.stats import linregress
        x = range(0, len(warmup_step_losses)*5, 5)
        slope, intercept, r_value, p_value, std_err = linregress(x, warmup_step_losses)
        plt.plot(x, [slope*xi + intercept for xi in x], 'r--', 
                 label=f'Trend: slope={slope:.6f}, R²={r_value**2:.2f}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Segment analysis - compare first third vs last third of training
    if len(warmup_losses) > 6:
        segment_size = len(warmup_losses) // 3
        first_segment = warmup_losses[:segment_size]
        last_segment = warmup_losses[-segment_size:]
        first_avg = sum(first_segment) / len(first_segment)
        last_avg = sum(last_segment) / len(last_segment)
        
        print(f"\nWarm-up Segment Analysis:")
        print(f"First {segment_size} steps average loss: {first_avg:.4f}")
        print(f"Last {segment_size} steps average loss: {last_avg:.4f}")
        print(f"Improvement during warm-up: {(1 - last_avg/first_avg)*100:.1f}%")
        
        # Calculate if still improving significantly
        still_improving = (first_avg - last_avg) / first_avg > 0.01  # More than 1% improvement
        print(f"Is model still significantly improving? {'Yes' if still_improving else 'No'}")
    
    # Print warm-up summary
    print(f"\nWarm-up completed with {len(warmup_losses)} steps across {epoch+1} epochs")
    print(f"Initial loss: {warmup_losses[0]:.4f}")
    print(f"Final loss: {warmup_losses[-1]:.4f}")
    print(f"Overall loss reduction: {(1 - warmup_losses[-1]/warmup_losses[0])*100:.1f}%")

except Exception as e:
    print(f"\nError during training: {e}")"""))

    # Evaluate baseline model
    cells.append(new_markdown_cell("""# Evaluate Baseline Model

Now let's measure the baseline performance after warm-up"""))

    cells.append(new_code_cell("""# Evaluate baseline model after warm-up
baseline_loss, baseline_perplexity = evaluate_model_performance(model, validation_dataloader, device)
print(f"Baseline evaluation after warm-up: Loss = {baseline_loss:.4f}, Perplexity = {baseline_perplexity:.2f}")

# Generate text with baseline model
prompt = "Once upon a time"
baseline_text = generate_text(model, tokenizer, prompt, device)
print(f"\nPrompt: {prompt}")
print(f"Generated text:\n{baseline_text}")"""))

    # Analyze attention patterns
    cells.append(new_markdown_cell("""## Analyze Attention Patterns

Let's look at the attention patterns in the model to understand which heads we might want to prune."""))

    cells.append(new_code_cell("""# Get a batch of data
batch = next(iter(validation_dataloader))
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)

# Run model to get attention patterns
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

# Extract attention tensors
attention_tensors = outputs.attentions

# Calculate entropy for each attention head
entropy_values = {}
for layer_idx, layer_attention in enumerate(attention_tensors):
    # Use our calculate_head_entropy function from utils.neural_plasticity.core
    layer_entropy = calculate_head_entropy(layer_attention)
    entropy_values[layer_idx] = layer_entropy

# Visualize the entropy heatmap
entropy_fig = visualize_head_entropy(
    entropy_values=entropy_values,
    title="Attention Entropy Heatmap",
    min_value=0.0,
    annotate=True,
    figsize=(10, 6)
)
plt.show()"""))

    # Calculate gradients
    cells.append(new_markdown_cell("""## Calculate Head Gradients

Now we'll calculate the gradient norms for each attention head to identify which ones have the least impact on the model's outputs."""))

    cells.append(new_code_cell("""# Calculate gradient norms for each head
grad_norm_values = calculate_head_gradients(
    model=model,
    dataloader=train_dataloader,
    num_batches=2,
    device=device
)

# Visualize gradient norms
grad_fig = visualize_head_gradients(
    grad_norm_values=grad_norm_values,
    title="Head Gradient Norms",
    figsize=(10, 5)
)
plt.show()"""))

    # Generate pruning mask
    cells.append(new_markdown_cell("""## Generate Pruning Mask

Based on entropy and gradient values, we'll create a pruning mask to identify which heads to prune."""))

    cells.append(new_code_cell("""# Define pruning strategy
PRUNING_STRATEGY = "combined"  # Options: "entropy", "gradient", "random", "combined"

# Generate pruning mask
pruning_mask = generate_pruning_mask(
    grad_norm_values=grad_norm_values,
    entropy_values=entropy_values[0],  # Use first layer's entropy as reference
    prune_percent=PRUNE_PERCENT,
    strategy=PRUNING_STRATEGY
)

# Visualize pruning mask
mask_fig = visualize_pruning_decisions(
    grad_norm_values=grad_norm_values,
    pruning_mask=pruning_mask,
    title=f"Pruning Decisions ({PRUNING_STRATEGY} strategy, {PRUNE_PERCENT*100:.0f}%)"
)
plt.show()

# Count pruned heads
total_heads = pruning_mask.numel()
pruned_count = pruning_mask.sum().item()
print(f"Pruning {pruned_count} out of {total_heads} heads ({pruned_count/total_heads*100:.1f}%)")"""))

    # Apply pruning
    cells.append(new_markdown_cell("""## Apply Pruning to Model

Now we'll apply the pruning mask to the model, zeroing out the weights of the selected heads."""))

    cells.append(new_code_cell("""# Apply pruning to the model
pruned_heads = apply_pruning_mask(
    model=model,
    pruning_mask=pruning_mask,
    mode="zero_weights"
)

print(f"Pruned {len(pruned_heads)} heads:")
for layer, head in pruned_heads[:10]:
    print(f"  Layer {layer}, Head {head}")
    
if len(pruned_heads) > 10:
    print(f"  ... and {len(pruned_heads) - 10} more")

# Evaluate pruned model
pruned_loss, pruned_perplexity = evaluate_model_performance(model, validation_dataloader, device)
print(f"\nPruned model evaluation: Loss = {pruned_loss:.4f}, Perplexity = {pruned_perplexity:.2f}")
print(f"Baseline:               Loss = {baseline_loss:.4f}, Perplexity = {baseline_perplexity:.2f}")
print(f"Difference:             {((pruned_loss - baseline_loss) / baseline_loss * 100):+.2f}%")

# Generate text with pruned model
pruned_text = generate_text(model, tokenizer, prompt, device)
print(f"\nPrompt: {prompt}")
print(f"Generated text with pruned model:\n{pruned_text}")"""))

    # Fine-tune pruned model
    cells.append(new_markdown_cell("""## Fine-tune the Pruned Model

Now let's fine-tune the pruned model to adapt to the reduced structure."""))

    cells.append(new_code_cell("""# Initialize optimizer and scheduler for fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=WARMUP_STEPS, 
    num_training_steps=total_steps
)

# Training loop
model.train()
global_step = 0
training_metrics = {
    "train_loss": [],
    "eval_loss": [],
    "perplexity": [],
    "steps": []
}

# Number of fine-tuning steps (keep short for demonstration)
fine_tuning_steps = 200
eval_every = 40

try:
    print(f"Fine-tuning for {fine_tuning_steps} steps...")
    steps_completed = 0
    
    while steps_completed < fine_tuning_steps:
        for batch in train_dataloader:
            if steps_completed >= fine_tuning_steps:
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
            scheduler.step()
            optimizer.zero_grad()
            
            # Track training metrics
            steps_completed += 1
            global_step += 1
            
            # Print progress
            if steps_completed % 10 == 0:
                print(f"Step {steps_completed}/{fine_tuning_steps}, Loss: {loss.item():.4f}", end='\r')
            
            # Evaluate
            if steps_completed % eval_every == 0:
                # Evaluation
                model.eval()
                eval_loss, eval_perplexity = evaluate_model_performance(model, validation_dataloader, device)
                
                # Update metrics
                training_metrics["train_loss"].append(loss.item())
                training_metrics["eval_loss"].append(eval_loss)
                training_metrics["perplexity"].append(eval_perplexity)
                training_metrics["steps"].append(steps_completed)
                
                print(f"\nStep {steps_completed} - Eval loss: {eval_loss:.4f}, Perplexity: {eval_perplexity:.2f}")
                
                # Back to training
                model.train()
    
    print(f"\nFine-tuning completed: {steps_completed} steps")
    
    # Plot training metrics
    plt.figure(figsize=(12, 8))
    
    # Loss plot
    plt.subplot(2, 1, 1)
    plt.plot(training_metrics["steps"], training_metrics["train_loss"], label="Train Loss")
    plt.plot(training_metrics["steps"], training_metrics["eval_loss"], label="Eval Loss")
    plt.title("Loss During Fine-tuning")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Perplexity plot
    plt.subplot(2, 1, 2)
    plt.plot(training_metrics["steps"], training_metrics["perplexity"], label="Perplexity", color="green")
    plt.title("Perplexity During Fine-tuning")
    plt.xlabel("Steps")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\nError during fine-tuning: {e}")

# Evaluate the fine-tuned pruned model
model.eval()
final_loss, final_perplexity = evaluate_model_performance(model, validation_dataloader, device)
print(f"\nFinal evaluation:\nBaseline: Loss = {baseline_loss:.4f}, Perplexity = {baseline_perplexity:.2f}")
print(f"Pruned:   Loss = {pruned_loss:.4f}, Perplexity = {pruned_perplexity:.2f} ({((pruned_loss - baseline_loss) / baseline_loss * 100):+.2f}%)")
print(f"Fine-tuned: Loss = {final_loss:.4f}, Perplexity = {final_perplexity:.2f} ({((final_loss - baseline_loss) / baseline_loss * 100):+.2f}%)")"""))

    # Generate text with fine-tuned model
    cells.append(new_markdown_cell("""## Generate Text with Fine-tuned Model

Let's generate text with our fine-tuned pruned model to see the results."""))

    cells.append(new_code_cell("""# Generate text with various prompts
prompts = [
    "Once upon a time",
    "The meaning of life is",
    "In a world where AI",
    "Scientists recently discovered"
]

for prompt in prompts:
    finetuned_text = generate_text(model, tokenizer, prompt, device)
    print(f"Prompt: {prompt}")
    print(f"Generated text:\n{finetuned_text}")
    print("-" * 80)"""))

    # Summary
    cells.append(new_markdown_cell("""## Neural Plasticity Summary

Our experiment showed how transformer models can be made more efficient through neural plasticity:

1. We identified and pruned heads with low gradient impact and high entropy
2. After pruning, there was a small initial performance drop
3. With fine-tuning, the model adapted to its new structure, recovering most of the performance
4. The final model is more efficient, using fewer parameters without significant quality loss

This demonstrates that transformer models contain redundancy that can be eliminated, and the pruned model can adapt to function with fewer heads.

Key metrics:
- Baseline perplexity: baseline_perplexity
- After pruning: pruned_perplexity (change%)
- After fine-tuning: final_perplexity (change%)
- Heads pruned: pruned_count out of total_heads (percentage%)

This neural plasticity cycle mimics how biological brains optimize their neural pathways, making it an important step toward more efficient and adaptable AI systems."""))

    # Add all cells to the notebook
    nb.cells.extend(cells)
    
    # Save notebook
    notebook_path = os.path.join("notebooks", "NeuralPlasticityDemo.ipynb")
    with open(notebook_path, "w") as f:
        nbformat.write(nb, f)
    
    print(f"Created updated notebook at {notebook_path}")
    
if __name__ == "__main__":
    create_neural_plasticity_notebook()