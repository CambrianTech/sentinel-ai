# Sentinel AI Data Module (sdata)

This module provides data loading, processing, and evaluation functionality for the Sentinel AI project.

## Background

This module was renamed from `sentinel_data` to `sdata` to avoid import conflicts with the Hugging Face datasets library. The datasets library tries to import a module called `sentinel_data.table`, which conflicts with our own module name.

## Usage

Import the module using:

```python
from sdata import load_dataset, prepare_dataset, evaluate_model, calculate_perplexity
```

## Components

- `dataset_loader.py`: Provides dataset loading and preparation functionality
- `eval.py`: Provides model evaluation functionality

## Dataset Loader

The dataset loader provides:

- `load_dataset()`: Load and prepare datasets for training and evaluation
- `prepare_dataset()`: Alias for load_dataset to maintain API compatibility

## Evaluation

The evaluation module provides:

- `evaluate_model()`: Evaluate model on a dataset
- `calculate_perplexity()`: Calculate perplexity on a text
- `load_eval_prompts()`: Load evaluation prompts for the specified dataset