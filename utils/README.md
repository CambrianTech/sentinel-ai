# Utilities

This directory contains utility functions and helper classes that support the Sentinel-AI framework.

## Key Components

### Core Utilities

- **`utils.py`**: Common helper functions used throughout the codebase
- **`train_utils.py`**: Specialized utilities for the training process
- **`checkpoint.py`**: Functions for saving and loading model checkpoints
- **`metrics_logger.py`**: Logging infrastructure for training and evaluation metrics
- **`progress_tracker.py`**: Tracks and visualizes training progress

### Model Wrappers

- **`model_wrapper.py`**: General wrapper for transformer models
- **`generation_wrapper.py`**: Specialized wrapper for text generation tasks

### Dynamic Architecture Tools

- **`dynamic_architecture.py`**: Functions for managing dynamic model architectures
- **`head_metrics.py`**: Tools for computing attention head metrics

### Metrics and Analysis

- **`metrics.py`**: Evaluation metrics for model performance
- **`head_metrics.py`**: Specific metrics for analyzing attention head behavior

## Usage Examples

### Checkpointing

```python
from utils.checkpoint import save_checkpoint, load_checkpoint

# Save model state
save_checkpoint(
    path="checkpoints/model.pth",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch
)

# Load model state
load_checkpoint(
    path="checkpoints/model.pth",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler
)
```

### Text Generation

```python
from utils.generation_wrapper import generate_text

# Generate text with the model
output = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_length=100,
    temperature=0.7,
    top_p=0.9
)
```

### Metrics Collection

```python
from utils.metrics import compute_perplexity
from utils.head_metrics import compute_attention_entropy

# Compute model perplexity
ppl = compute_perplexity(model, eval_dataloader)

# Compute attention head entropy
entropy = compute_attention_entropy(attention_weights)
```