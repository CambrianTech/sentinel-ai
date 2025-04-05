# Entropy Heatmap Figure

The following figure visualizes the distribution of attention entropy across layers and heads in a transformer model. This visualization helps identify patterns in how attention focus is distributed throughout the model and highlights potential candidates for entropy-based pruning.

```
Figure: Attention Entropy Distribution Across a GPT-2 Model

   ┌────────────────────────────────────────────┐
H  │                                            │
e  │  [ Layer x Head Heatmap ]                  │
a  │                                            │
d  │  Low Entropy                   High Entropy│
   │  (Focused)                     (Diffuse)   │
I  │  ┌─────────────────────────────────────┐  │
n  │  │                                     │  │
d  │  │                                     │  │
e  │  │                                     │  │
x  │  │                                     │  │
   │  │                                     │  │
   │  └─────────────────────────────────────┘  │
   │    Layer 0       Layer Index       Layer N │
   │                                            │
   │  ┌─────────────────────────────────────┐  │
   │  │ Color Scale:                        │  │
   │  │ ████ < 0.5  ████ 0.5-1.0 ████ 1.0-1.5│  │
   │  │ ████ 1.5-2.0 ████ 2.0-2.5 ████ > 2.5 │  │
   │  └─────────────────────────────────────┘  │
   └────────────────────────────────────────────┘
```

## Figure Description

The heatmap displays attention entropy values across all layers and heads in the model. Each cell represents a specific attention head's entropy value, with color intensity indicating the degree of entropy:

- **Darker colors (lower entropy)**: Indicate heads with more focused attention patterns that typically attend strongly to specific tokens or positions
- **Lighter colors (higher entropy)**: Indicate heads with more diffuse attention patterns that distribute attention more uniformly across tokens

## Key Observations

Several notable patterns typically emerge in these entropy visualizations:

1. **Layer-wise Progression**: Later layers often show more specialized (lower entropy) attention patterns as they build on earlier representations
2. **Functional Clustering**: Heads with similar functions often exhibit similar entropy profiles
3. **Positional Heads**: Heads that primarily attend to fixed positions (e.g., adjacent tokens) usually have very low entropy
4. **Global Context Heads**: Heads that integrate information from the entire sequence tend to have higher entropy

## Experimental Relevance

This visualization directly informs entropy-based pruning by identifying:

- High-entropy heads as primary pruning candidates
- Patterns of redundancy across the model
- Potential "specialist" heads that should be preserved

When combined with performance metrics after pruning, this visualization helps identify correlations between attention entropy and functional importance, providing insights into how transformers allocate and organize information processing capabilities.

## Implementation Notes

The entropy values are calculated using the equation:

$$H(h) = \frac{1}{B \cdot S} \sum_{b=1}^{B} \sum_{i=1}^{S} -\sum_{j=1}^{S} A^{(h)}_{b,i,j} \log A^{(h)}_{b,i,j}$$

Where:
- $A^{(h)}_{b,i,j}$ is the attention probability from token $i$ to token $j$ for head $h$ in batch $b$
- $B$ is the batch size and $S$ is the sequence length

For practical visualization, we typically average entropy values across a representative sample of input text from the target domain.