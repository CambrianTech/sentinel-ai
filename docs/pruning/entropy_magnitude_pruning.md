# Scientific Pruning Strategies: Entropy and Magnitude-Based Approaches

This document describes the implementation and usage of scientific pruning strategies for transformer models in the Sentinel-AI project.

## Introduction

Pruning attention heads in transformer models can reduce computational requirements while maintaining model performance. The Sentinel-AI project implements two scientifically-backed pruning strategies:

1. **Entropy-based Pruning**: Identifies and prunes heads with high entropy in their attention distributions, which typically correspond to heads that don't focus sharply on specific tokens.

2. **Magnitude-based Pruning**: Prunes heads with the lowest weight magnitudes, assuming that smaller weights contribute less to the model's output.

## Implementation

The implementation can be found in `sentinel/pruning/entropy_magnitude.py`. The key functions are:

### Entropy-Based Pruning

```python
from sentinel.pruning.entropy_magnitude import collect_attention_distributions, entropy_based_pruning

# Collect attention distributions from a few batches
distributions = collect_attention_distributions(
    model,
    dataloader,
    num_batches=5
)

# Apply entropy-based pruning
pruned_heads = entropy_based_pruning(
    model,
    distributions,
    prune_ratio=0.3,  # Prune 30% of heads
    safe_update_tensor_fn=safe_update_tensor
)
```

### Magnitude-Based Pruning

```python
from sentinel.pruning.entropy_magnitude import magnitude_based_pruning

# Apply magnitude-based pruning
pruned_heads = magnitude_based_pruning(
    model,
    prune_ratio=0.3,  # Prune 30% of heads
    safe_update_tensor_fn=safe_update_tensor
)
```

### Helper Functions

The module also provides helper functions:

- `compute_attention_entropy`: Computes the entropy of attention distributions
- `update_mask`: Safely updates tensors without breaking gradients
- `_apply_pruning`: Internal function to apply pruning to specified heads

## Integration with the Benchmark System

The entropy and magnitude-based pruning strategies are integrated with the benchmark system in `scripts/benchmark_with_metrics.py`. The benchmark script includes:

- Support for multiple pruning strategies, including entropy and magnitude
- Evaluation before and after pruning to measure performance impact
- Fine-tuning after pruning to recover performance
- Comprehensive metrics collection and visualization

To run a benchmark with these strategies:

```bash
python scripts/benchmark_with_metrics.py \
  --model_name distilgpt2 \
  --pruning_strategies "entropy,magnitude,random" \
  --pruning_levels "0.1,0.3,0.5" \
  --learning_steps 100 \
  --eval_dataset gutenberg \
  --use_real_data
```

## Testing

Unit tests for the pruning implementations can be found in `tests/unit/pruning/test_entropy_magnitude.py`. A simplified integration test is available in `scripts/test_entropy_magnitude_pruning.py`.

To run the tests:

```bash
# Run unit tests
python -m unittest tests/unit/pruning/test_entropy_magnitude.py

# Run the integration test
python scripts/test_entropy_magnitude_pruning.py --model_name distilgpt2
```

## Technical Details

### Entropy Calculation

Entropy is calculated as:

$H(p) = -\sum_{i} p_i \log(p_i)$

where $p_i$ is the attention probability for token $i$. Higher entropy indicates more uniform attention, while lower entropy indicates more focused attention.

### Magnitude Calculation

Magnitude is calculated as the norm of all weight matrices associated with a head:

$M = ||W_Q|| + ||W_K|| + ||W_V|| + ||W_O||$

where $W_Q$, $W_K$, $W_V$ are the query, key, and value projection weights for the head, and $W_O$ is the output projection weights.

## Compatibility

The implementation handles different model architectures:

- GPT-2 style (with `c_attn` and `c_proj` weights)
- BERT/RoBERTa style (with separate `q_proj`, `k_proj`, `v_proj`, and `o_proj` weights)
- Generic fallback for other architectures

## Planned Improvements

Future enhancements to these pruning strategies include:

1. Support for more model architectures (T5, OPT, etc.)
2. Hybrid strategies that combine entropy and magnitude metrics
3. Dynamic pruning during training
4. Consideration of head interactions and redundancy
5. Regularization-based pruning approaches