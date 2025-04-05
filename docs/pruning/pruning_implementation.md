# Scientific Pruning Implementation

## Architecture Overview

The scientific pruning system in Sentinel-AI provides a rigorous approach to identifying and pruning attention heads in transformer models. The implementation supports multiple pruning strategies and accommodates various model architectures through dynamic detection and adaptation.

## Core Components

### 1. Architecture Detection

The system automatically identifies the model structure and adapts to different transformer implementations:

- **GPT-style models**: Models with unified QKV projections (`c_attn` and `c_proj`)
- **BERT-style models**: Models with separate query, key, value projections (`q_proj`, `k_proj`, `v_proj`)
- **Sentinel-AI Adaptive models**: Models with the blocks-based architecture specific to Sentinel-AI

This architecture-aware approach ensures compatibility across model families and enables consistent pruning behavior regardless of the underlying implementation.

### 2. Pruning Strategies

The system implements two complementary, scientifically-grounded pruning strategies:

- **[Entropy-based pruning](methods_entropy.md)**: Measures the information-theoretic entropy of attention distributions to identify unfocused heads
- **[Magnitude-based pruning](methods_magnitude.md)**: Analyzes weight norms to identify heads with minimal contribution to the model

Each strategy provides a different perspective on head importance, allowing for comparative analysis and optimized pruning based on specific requirements.

### 3. Safe Pruning Application

Once heads are selected for pruning, the system applies pruning through a flexible mechanism that adapts to the model's structure:

- **Detection of pruning interfaces**:
  - `gate` parameter (common in adaptive models)
  - `head_mask` attribute (common in HuggingFace models)
  - `pruning_mask` attribute
  - Dynamic creation of mask buffers for models without built-in pruning support

- **Safe tensor updates**:
  - Preserves gradient flow for remaining heads
  - Ensures compatibility with subsequent fine-tuning
  - Handles in-place operations safely

### 4. Integration Points

The pruning system integrates with other components of Sentinel-AI:

- **Fine-tuning**: Seamless transition to post-pruning fine-tuning
- **Metrics collection**: Recording of pruning decisions and their impact
- **Visualization**: Support for visualizing pruning patterns and head importance

## Implementation Highlights

### Model-Agnostic Design

```python
# For adaptive transformer in Sentinel-AI, the blocks are accessed differently
if hasattr(model, 'blocks'):
    # Direct access to blocks
    return _compute_magnitude_from_blocks(model.blocks, prune_ratio, safe_update_tensor_fn)

# Try standard HuggingFace model structures
transformer = None
if hasattr(model, 'transformer'):
    transformer = model.transformer
elif hasattr(model, 'model') and hasattr(model.model, 'transformer'):
    transformer = model.model.transformer
elif hasattr(model, 'base_model') and hasattr(model.base_model, 'transformer'):
    transformer = model.base_model.transformer
```

### Hook-Based Attention Collection

```python
# Register hooks for each attention layer
hooks = []
for i, layer in enumerate(transformer.h):
    hook = layer.attn.register_forward_hook(hook_fn(i))
    hooks.append(hook)

# Process batches with hooks active
with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        # ...
        
# Remove hooks after collection
for hook in hooks:
    hook.remove()
```

### Architecture-Specific Weight Access

```python
# GPT-2 style
if hasattr(layer.attn, 'c_attn'):
    qkv_weight = layer.attn.c_attn.weight
    proj_weight = layer.attn.c_proj.weight
    # ...
    
# BERT/RoBERTa style
elif hasattr(layer.attn, 'q_proj'):
    q_weight = layer.attn.q_proj.weight
    k_weight = layer.attn.k_proj.weight
    # ...
```

### Flexible Pruning Application

```python
# Check for mask_heads method (HuggingFace compatible)
if hasattr(layer.attn, 'mask_heads'):
    layer.attn.mask_heads(head_indices)
    
# Check for head_mask attribute
elif hasattr(layer.attn, 'head_mask'):
    update_mask(layer.attn.head_mask, head_indices, 0.0, safe_update_tensor_fn)
    
# Check for gate parameter
elif hasattr(layer.attn, 'gate'):
    update_mask(layer.attn.gate, head_indices, 0.0, safe_update_tensor_fn)
```

## Usage Examples

### Entropy-Based Pruning

```python
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
# Apply magnitude-based pruning
pruned_heads = magnitude_based_pruning(
    model,
    prune_ratio=0.3,  # Prune 30% of heads
    safe_update_tensor_fn=safe_update_tensor
)
```

## Scientific Applications

The implementation enables several scientific investigations:

1. **Comparative analysis** of different pruning strategies
2. **Layer-wise pruning patterns** and their impact on model behavior
3. **Correlation studies** between entropy, magnitude, and functional importance
4. **Task-specific pruning** analysis across different domains

By combining these scientific pruning methods with Sentinel-AI's adaptive architecture, researchers can gain deeper insights into transformer model behavior and optimize model efficiency without sacrificing performance.