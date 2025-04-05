# Magnitude-Based Head Pruning

## Overview

Magnitude-based pruning assesses the importance of attention heads by measuring the weight magnitudes associated with each head. This method operates on the principle that smaller weight norms indicate less influential heads, making them suitable candidates for pruning without significant impact on model performance.

## Mathematical Formulation

For each attention head $h$ in layer $l$, we compute a magnitude score $M(l,h)$ as:

$$M(l,h) = \|W^Q_{l,h}\|_F + \|W^K_{l,h}\|_F + \|W^V_{l,h}\|_F + \|W^O_{l,h}\|_F$$

where:
- $W^Q_{l,h}$, $W^K_{l,h}$, $W^V_{l,h}$, and $W^O_{l,h}$ are the query, key, value, and output projection weights for head $h$ in layer $l$
- $\|\cdot\|_F$ denotes the Frobenius norm (essentially the square root of the sum of squared elements)

## Implementation Details

The magnitude-based pruning algorithm consists of the following steps:

1. **Weight Extraction**:
   - Identify and access the weight matrices for each attention head
   - Handle different model architectures (GPT-style, BERT-style, etc.) with appropriate weight access patterns

2. **Magnitude Computation**:
   - For GPT-style models with unified QKV projections:
     - Slice the appropriate sections of the combined weight matrix
     - Calculate norm for each head's query, key, value, and output projection weights
   
   - For BERT-style models with separate Q/K/V projections:
     - Access individual weight matrices
     - Calculate norm for each head's portion of these matrices

3. **Head Ranking**:
   - Sort heads by their magnitude scores in ascending order (lowest magnitude first)
   - Heads with the smallest magnitude scores are considered less important

4. **Pruning Application**:
   - Select bottom N heads based on the specified pruning ratio
   - Apply pruning by setting gate values or masks to zero
   - Use safe tensor update mechanisms to preserve gradient flow

## Code Excerpt

```python
# For GPT-2 style models
# Compute magnitude for each head
for h in range(num_heads):
    # Calculate offsets for Q, K, V
    q_start = h * head_dim
    q_end = (h + 1) * head_dim
    k_start = q_start + num_heads * head_dim
    k_end = q_end + num_heads * head_dim
    v_start = k_start + num_heads * head_dim
    v_end = k_end + num_heads * head_dim
    
    # Extract weights for this head
    q = qkv_weight[q_start:q_end]
    k = qkv_weight[k_start:k_end]
    v = qkv_weight[v_start:v_end]
    o = proj_weight[:, q_start:q_end]
    
    # Compute magnitude as norm of all weights
    magnitude = (q.norm() + k.norm() + v.norm() + o.norm())
    magnitude_scores.append((i, h, magnitude.item()))
```

## Scientific Rationale

The magnitude-based approach is supported by several theoretical principles:

1. **Weight Decay Effect**: During training, less important weights naturally tend toward smaller magnitudes due to weight decay regularization.
   
2. **Network Pruning Theory**: Research in neural network pruning suggests that parameters with small magnitudes often contribute less to the model's output.
   
3. **Sensitivity Approximation**: Weight magnitudes can serve as a first-order approximation of the network's sensitivity to pruning specific components.

By removing heads with small weight magnitudes, we target components that likely have minimal impact on the model's predictions, enabling efficient pruning with minimal performance degradation.

## Advantages

- **Computationally efficient**: Requires no forward passes, making it faster than methods requiring inference
- **Data-independent**: Can be applied without access to any dataset
- **Simple implementation**: Straightforward to compute across various model architectures
- **Theoretically grounded**: Aligns with established principles from network pruning literature

## Limitations

- **Static analysis**: Does not consider actual model behavior on data
- **Potential false positives**: Some small-magnitude heads might still be functionally important
- **Architecture-dependent**: Weight structures vary across model families, requiring specific implementation details
- **Scale sensitivity**: May be affected by initialization schemes and training hyperparameters