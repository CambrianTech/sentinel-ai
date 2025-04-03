# Optimized Model Implementations

This directory contains optimized implementations of the Sentinel-AI transformer model. The optimizations focus on several key areas:

1. **Attention Mechanism**: Optimized attention with specialized code paths for pruned heads
2. **Integration Points**: Improved interactions between components to minimize overhead
3. **CPU/GPU Specializations**: Device-specific optimizations for better performance
4. **Memory Efficiency**: Reduced memory usage and better caching strategies

## Optimization Levels

The optimized implementation supports multiple optimization levels:

| Level | Description | Best For |
|-------|-------------|----------|
| 0 | No optimizations | Debugging, baseline comparisons |
| 1 | Default optimizations | General use, balance of features and performance |
| 2 | Aggressive optimizations | CPU inference, production deployment |  
| 3 | Extreme optimizations | GPU with heavy pruning, maximum throughput |

## Performance Recommendations

Based on our profiling and validation results:

- For **maximum throughput**:
  - Use the original model with 70% pruning (fastest option)
  - Up to 28 tokens/sec on CPU

- For **best CPU performance with agency features**:
  - Use optimization level 2
  - Apply 30-50% pruning with the optimized model
  - Disable baseline integration
  - Around 19-20 tokens/sec on CPU

- For **best balance of quality and speed**:
  - Use optimization level 2
  - Apply 30% pruning
  - Enable baseline integration on GPU

Our testing confirms that while the optimized model provides important agency features, the original model with heavy pruning remains the fastest option for pure throughput. The optimized implementation still shows performance improvements over the non-pruned baseline.

## Profiling

You can profile the model with different configurations using the profiling tools:

```bash
# Basic profiling with default settings
python scripts/profile_full_model.py --model_name gpt2 --device cpu

# Test all optimization levels
python scripts/profile_full_model.py --model_name gpt2 --device cpu --test_all_optimization_levels

# Compare optimization approaches
python scripts/profile_full_model.py --model_name gpt2 --device cpu --test_integration_points 

# Export results to various formats
python scripts/export_profiling_results.py --input_file profiling_results/full_model/your_results.json --format all --create_charts
```

## Integration Guide

When integrating the optimized model, you can set the optimization level:

```python
# Set optimization level via environment variable
import os
os.environ["OPTIMIZATION_LEVEL"] = "2"  # For best CPU performance

# Or at model creation time
model = load_adaptive_model(
    baseline_model=baseline_model,
    optimization_level=2
)
```

See the docstring in `integration_optimized.py` for detailed performance insights.