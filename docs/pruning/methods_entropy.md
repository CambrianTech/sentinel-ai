# Entropy-Based Head Pruning

## Overview

Entropy-based pruning identifies attention heads with diffuse attention patterns by directly measuring the information-theoretic entropy of attention distributions. This method targets heads that exhibit unfocused attention behavior, operating under the hypothesis that heads with highly entropic attention patterns contribute less meaningful information to the model's output.

## Mathematical Formulation

For each attention head $h$, we compute entropy $H(h)$ across a representative subset of the data:

$$H(h) = \frac{1}{B \cdot S} \sum_{b=1}^{B} \sum_{i=1}^{S} -\sum_{j=1}^{S} A^{(h)}_{b,i,j} \log A^{(h)}_{b,i,j}$$

where:
- $A^{(h)}_{b,i,j}$ is the attention probability from token $i$ to token $j$ for head $h$ in batch $b$
- $B$ and $S$ are the batch size and sequence length, respectively

## Implementation Details

The entropy-based pruning algorithm proceeds through the following steps:

1. **Attention Distribution Collection**:
   - Register forward hooks on attention modules to capture attention probabilities
   - Process multiple batches of input data to gather a representative sample
   - Store attention maps for each layer and head

2. **Entropy Computation**:
   - Calculate entropy for each head using the formula above
   - Add a small epsilon ($\approx 10^{-8}$) to prevent numerical instability in log computation

3. **Head Ranking**:
   - Sort heads by their entropy scores in descending order (highest entropy first)
   - Heads with the highest entropy values exhibit the least focused attention patterns
   - These heads are considered primary candidates for pruning

4. **Pruning Application**:
   - Select top N heads based on the specified pruning ratio
   - Apply pruning by setting gate values or masks to zero
   - Use safe tensor update mechanisms to preserve gradient flow

## Code Excerpt

```python
def compute_attention_entropy(attn_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes entropy per attention head from the attention probability tensor.
    
    Args:
        attn_probs: Attention probability tensor with shape (batch_size, num_heads, seq_len, seq_len)
        eps: Small constant to avoid log(0)
        
    Returns:
        Tensor of shape (num_heads) containing the average entropy for each head
    """
    log_attn = torch.log(attn_probs + eps)
    entropy = -torch.sum(attn_probs * log_attn, dim=-1)  # shape: (batch_size, num_heads, seq_len)
    return entropy.mean(dim=(0, 2))  # average over batch and sequence
```

## Scientific Rationale

Heads with high entropy distribute attention broadly across all tokens rather than focusing on specific informative tokens. This suggests that:

1. The head is not learning meaningful token relationships
2. The information captured is redundant with other heads
3. The head may be capturing noise rather than signal

By pruning high-entropy heads, we remove components that contribute minimal information to the model's output, potentially improving inference efficiency without sacrificing performance.

## Advantages

- **Requires no labels**: Based only on attention distributions, not task performance
- **Interpretable**: Provides a clear information-theoretic justification
- **Data-informed**: Uses actual model behavior on real data
- **Architecture-agnostic**: Works across different transformer model families

## Limitations

- **Computational overhead**: Requires forward passes to collect attention distributions
- **Potentially task-dependent**: Entropy patterns may vary across different input domains
- **Ignores head interactions**: Does not account for potential complementary relationships between heads