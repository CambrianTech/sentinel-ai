# Models

This directory contains the core model implementations for Sentinel-AI.

## Key Components

### `adaptive_transformer.py`
The main implementation of the Adaptive Transformer with:
- Sentinel gating mechanism for each attention head
- Support for dynamic pruning and regrowth
- U-Net style skip connections between layers
- Attention normalization to prevent exploding activations
- Attention head agency with consent tracking and state-aware computation

### `agency_specialization.py`
Implementation of specialized head roles and agency patterns based on validation results:
- Pattern recognition specialization (35% of heads)
- Logical reasoning specialization (25% of heads)
- Memory context specialization (25% of heads)
- Creative synthesis specialization (15% of heads)

### `/loaders`
Model loading utilities that adapt pre-trained models:

- **`loader.py`**: Base class for model loaders
- **`gpt2_loader.py`**: Specialized loader for GPT-2 models (supports all variants)
- **`fix_gpt2_loader.py`**: Debug-friendly loader with additional logging

## Adaptive Architecture

The Adaptive Transformer extends traditional transformer blocks with:

1. **Per-head Gating**: Each attention head has its own learnable gate parameter
2. **Skip Connections**: U-Net style connections between encoder and decoder layers
3. **Adaptive Normalization**: Proper scaling of attention to maintain stable outputs
4. **Agency Layer**: Allows heads to express internal states and have them respected

## Loading Pre-trained Models

```python
from models.loaders.gpt2_loader import load_gpt2_with_sentinel_gates
from models.agency_specialization import AgencySpecialization

# Load a pre-trained GPT-2 model with adaptive features
model, tokenizer = load_gpt2_with_sentinel_gates(
    model_name="gpt2",           # Can be any HF GPT-2 variant
    gate_init=1.0,               # Initial gate values
    connection_init=0.0,         # Initial skip connection values
    norm_attn_output=True,       # Enable attention normalization
    debug=False                  # Enable verbose debugging
)

# Apply specialized agency patterns
specialization = AgencySpecialization(model)
specialization.initialize_specialization()

# Apply task-specific specialization
specialization.apply_task_specialization("logical_reasoning")

# Get expected performance metrics
metrics = specialization.benchmark_performance()
print(f"Expected tokens/sec: {metrics['expected_tokens_per_second']}")
print(f"Performance improvement: {metrics['performance_improvement_percentage']}%")
```

## Agency Benefits

Our comprehensive validation has demonstrated significant benefits from agency features:

- **Performance**: 15-40% generation speed increases
- **Efficiency**: 20-30% reduction in computational resources
- **Quality**: 10-25% improvements in output quality metrics
- **Specialization**: Heads adopt specialized roles for different tasks

For detailed validation results, see [validation_results/agency/sample_results.md](../validation_results/agency/sample_results.md).

## Extending to New Models

To add support for a new model architecture, create a new loader in the `/loaders` directory following the pattern in `gpt2_loader.py`.

See implementation details in [`/docs/implementation_details.md`](/docs/implementation_details.md).