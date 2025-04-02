# Improved Fine-Tuner for Pruned Models

This document describes the `ImprovedFineTuner` implementation which addresses stability issues when fine-tuning large pruned language models, particularly OPT-1.3B and similar models that are prone to NaN loss and other training instabilities.

## Key Features

The `ImprovedFineTuner` provides several enhancements over the standard `FineTuner`:

1. **Enhanced Stability**
   - NaN detection and recovery with gradient clipping
   - Dynamic learning rate and batch size reduction when instabilities occur
   - Safe computation patterns with fallbacks for division by zero and other numerical issues
   - Memory-efficient operation for large models
   - Automatic garbage collection during training to prevent OOM errors

2. **Model-Specific Optimizations**
   - Specialized handling for different model architectures (OPT, GPT-2, BLOOM, etc.)
   - Architecture-specific learning rates
   - Special token handling customized for each model family

3. **Improved Dataset Handling**
   - Better dataset format detection
   - Handling for insufficient token sequences
   - More robust tokenization with padding/truncation
   - Support for various dataset configurations

4. **Comprehensive Logging**
   - Detailed logging of training progress
   - Training metrics visualization with NaN counts
   - Training dynamics monitoring

## Usage

### Basic Usage

```python
from utils.pruning import ImprovedFineTuner, PruningModule

# Initialize pruning module
pruning_module = PruningModule("facebook/opt-1.3b")
pruning_module.load_model()

# Apply pruning
# ... (pruning code) ...
pruned_params = pruning_strat.prune_heads(original_params, head_indices)

# Initialize improved fine-tuner
fine_tuner = ImprovedFineTuner(
    pruning_module, 
    dataset_name="wikitext",
    dataset_config="wikitext-103-v1",
    batch_size=4
)

# Fine-tune the pruned model
tuned_params, metrics = fine_tuner.fine_tune(
    pruned_params,
    num_epochs=2,
    learning_rate=1e-5,  # Lower learning rate recommended for larger models
    evaluate_interval=5
)

# Plot training progress
fine_tuner.plot_training_progress()
```

### Command Line Usage

```bash
# Test with OPT-350M model (medium size)
python scripts/test_improved_fine_tuner.py --model facebook/opt-350m --strategy entropy --pruning_level 0.3 --epochs 2

# Test with OPT-1.3B model (large model prone to instability)
python scripts/test_improved_fine_tuner.py --model facebook/opt-1.3b --use_improved --epochs 1 --batch_size 2

# Compare original vs improved fine-tuner
python scripts/test_improved_fine_tuner.py --model facebook/opt-350m --use_original --use_improved --epochs 1
```

## Implementation Details

### Automatic Batch Size Adjustment

```python
def _adjust_batch_size_for_model_size(self):
    """Adjust batch size based on model size to prevent OOM errors"""
    model_name = self.pruning_module.model_name.lower()
    
    # Detect large models
    if "1.3b" in model_name or "large" in model_name:
        # Reduce batch size for large models
        old_batch_size = self.batch_size
        self.batch_size = max(1, self.batch_size // 4)
        logger.info(f"Reduced batch size from {old_batch_size} to {self.batch_size} for large model {model_name}")
    elif "medium" in model_name or "base" in model_name or "350m" in model_name:
        # Reduce batch size for medium models
        old_batch_size = self.batch_size
        self.batch_size = max(1, self.batch_size // 2)
        logger.info(f"Reduced batch size from {old_batch_size} to {self.batch_size} for medium-sized model {model_name}")
```

### Model-Specific Forward Pass

```python
# Handle different model architectures
try:
    # Get logits from model - don't pass 'train' param for OPT models
    if self.is_opt_model:
        # OPT models don't accept 'train' parameter
        outputs = model(**batch, params=params)
    else:
        # Other models like GPT-2 might need the 'train' parameter
        outputs = model(**batch, params=params, train=True)
```

### NaN Detection and Recovery

```python
# Check for NaN loss
if jnp.isnan(loss).any() or jnp.isinf(loss).any():
    nan_count += 1
    logger.warning(f"NaN loss at step {total_steps}")
    
    # If we get too many NaNs, reduce batch size and learning rate
    if nan_count > 5:
        logger.warning("Too many NaN losses, reducing batch size and learning rate")
        old_batch_size = self.batch_size
        self.batch_size = max(1, self.batch_size // 2)
        if old_batch_size == self.batch_size:  # Can't reduce further
            logger.error("Cannot reduce batch size further, aborting fine-tuning")
            break
            
        # Create new dataset with smaller batch size
        dataset = self._prepare_dataset()
        
        # Create new train state with smaller learning rate
        learning_rate = learning_rate / 2
        logger.info(f"Reducing batch size to {self.batch_size} and learning rate to {learning_rate}")
        self.train_state = self._create_train_state(self.train_state.params, learning_rate)
```

### Safe Loss Computation

```python
# Apply mask and calculate mean - safely with jnp.where to avoid division by zero
masked_loss = loss * shift_mask
mask_sum = shift_mask.sum()

# Safe division with fallback
loss = jnp.where(
    mask_sum > 0,
    masked_loss.sum() / mask_sum,
    jnp.array(0.0)  # Default value if mask_sum is 0
)
```

## Best Practices

1. **Learning Rate Selection**
   - For large models (1B+ parameters): Use a learning rate of 1e-5 or lower
   - For medium models (350M-1B parameters): Use a learning rate of 2e-5
   - For small models (<350M parameters): A learning rate of 5e-5 is usually sufficient

2. **Batch Size Selection**
   - The ImprovedFineTuner will automatically adjust batch sizes based on model size
   - For manual adjustment, consider:
     - Large models (1B+): 1-2 examples per batch
     - Medium models (350M-1B): 2-4 examples per batch
     - Small models (<350M): 4-8 examples per batch

3. **GPU vs. CPU Training**
   - GPU: Best for models 350M+ parameters
   - CPU: Viable for models <350M parameters but will be slow
   - TPU: Supported and recommended for large models

4. **When to Use ImprovedFineTuner vs. Standard FineTuner**
   - Always use ImprovedFineTuner for:
     - OPT models of any size
     - Any model 1B+ parameters
     - Any case with NaN loss issues
   - Standard FineTuner can be used for:
     - GPT-2 family models (simple and stable)
     - Small models (<350M) when speed is prioritized over stability

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory (OOM) Errors**
   - Reduce batch size further (even to 1)
   - Reduce sequence length (max_seq_length parameter)
   - Use a smaller model if possible

2. **Persistent NaN Loss**
   - Further reduce learning rate (try 5e-6)
   - Try a different optimizer (AdamW instead of Adam)
   - Check for very small or zero values in attention masks

3. **Very Slow Training**
   - Ensure you're using GPU/TPU when available
   - Reduce dataset size for testing (train[:1000] split)
   - Use synthetic dataset for quick tests

### Getting Help

If you encounter persistent issues with the ImprovedFineTuner, check:
- The logs for specific error messages
- GPU memory utilization during training
- Model compatibility with the pruning approach

For further assistance, open an issue in the GitHub repository with detailed information about your setup and the specific error encountered.