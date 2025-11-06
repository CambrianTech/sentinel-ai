# Stability Utilities for Fine-Tuning

This module provides utilities to enhance the stability of fine-tuning language models,
particularly after pruning attention heads.

## Features

- NaN detection and prevention during training
- Safe loss computation to handle numerical instabilities
- Testing utilities to verify stability

## Usage

### Basic Usage

```python
from utils.pruning.stability import patch_fine_tuner

# Patch a fine-tuner instance with NaN-safe mechanisms
fine_tuner = patch_fine_tuner(fine_tuner, model_name="gpt2")
```

### Testing

You can test the NaN prevention mechanisms locally:

```bash
python -m utils.pruning.stability.test_nan_prevention --model gpt2 --verbose
```

## Implementation Details

The module works by wrapping the loss function with a NaN-safe version that:

1. Detects NaN values in inputs, intermediate calculations, and outputs
2. Replaces NaN values with safe defaults
3. Provides fallback mechanisms when calculations fail
4. Handles architecture-specific differences between models

This approach ensures that training can proceed even when numerical instabilities occur,
which is particularly important for fine-tuning large language models after pruning,
especially OPT models which have shown susceptibility to NaN issues.