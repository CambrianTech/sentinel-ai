# Attention Head Pruning Methodology

## Overview

Transformer-based language models have shown remarkable capabilities, but their computational requirements grow quadratically with sequence length. Our research demonstrates that many attention heads in these models contribute minimally to the output and can be pruned without significant performance degradation.

The Sentinel-AI framework introduces a dynamic, metric-driven approach to attention head pruning that:

1. Identifies underutilized or redundant attention heads
2. Strategically deactivates them to improve computational efficiency
3. Maintains the option to reactivate heads when needed for complex tasks

## Pruning Strategies

Our framework supports multiple pruning strategies, each targeting different aspects of attention head importance:

### Entropy-Based Pruning

Entropy-based pruning targets attention heads with high entropy in their attention distributions. When an attention head produces a flat or uniform distribution across all tokens (high entropy), it typically contributes less meaningful information to the model's output.

```python
def apply_entropy_pruning(model, pruning_level, metrics):
    # Get entropy from metrics
    entropy = metrics["entropy"]
    
    # Higher entropy = less focused attention = more likely to be pruned
    # Sort heads by entropy and prune those with highest values
    flat_entropy = entropy.view(-1)
    _, indices = torch.sort(flat_entropy, descending=True)
    indices_to_prune = indices[:heads_to_prune]
    
    # Apply pruning
    for idx in indices_to_prune:
        layer_idx = idx.item() // num_heads
        head_idx = idx.item() % num_heads
        model.blocks[layer_idx]["attn"].gate[head_idx] = 0.001  # Nearly zero
```

### Gradient-Based Pruning

Gradient-based pruning examines the magnitude of gradients flowing through each attention head. Heads with consistently small gradient norms have minimal impact on the loss function and can be pruned with minimal effect on model performance.

```python
def apply_gradient_pruning(model, pruning_level, metrics):
    # Get gradient norms
    grad_norm = metrics["grad_norm"]
    
    # Lower gradient norm = less important head = more likely to be pruned
    flat_grad_norm = grad_norm.view(-1)
    _, indices = torch.sort(flat_grad_norm)
    indices_to_prune = indices[:heads_to_prune]
    
    # Apply pruning
    for idx in indices_to_prune:
        layer_idx = idx.item() // num_heads
        head_idx = idx.item() % num_heads
        model.blocks[layer_idx]["attn"].gate[head_idx] = 0.001  # Nearly zero
```

### Random Pruning (Baseline)

As a control strategy, we implement random pruning which selects attention heads to prune without regard to their importance metrics. This serves as a baseline to evaluate the effectiveness of our metric-driven approaches.

```python
def apply_random_pruning(model, pruning_level):
    # Get a flattened list of (layer, head) tuples
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    
    # Randomly select heads to prune
    pruned_head_indices = np.random.choice(len(all_heads), heads_to_prune, replace=False)
    
    # Apply pruning
    for idx in pruned_head_indices:
        layer_idx, head_idx = all_heads[idx]
        model.blocks[layer_idx]["attn"].gate[head_idx] = 0.001  # Nearly zero
```

## Implementation Mechanism

The pruning mechanism in Sentinel-AI operates through learnable "gate" parameters assigned to each attention head. These gates modulate the contribution of each head to the network's output:

1. **Full Contribution**: Gate values near 1.0 allow the head to contribute normally
2. **Partial Contribution**: Intermediate values (e.g., 0.5) reduce the head's influence
3. **Pruned**: Gate values near 0.0 effectively disable the head

This gating mechanism is implemented as:

```python
# In the forward pass of attention:
attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, value)
attn_output = attn_output * self.gate.view(1, 1, -1, 1)  # Apply gate values
```

## Experimental Results

Our benchmark experiments demonstrate the effectiveness of strategic pruning compared to random pruning:

- Pruning up to 70% of attention heads maintains model performance
- Entropy-based pruning outperforms random pruning at high pruning levels
- Inference speed increases with entropy-based pruning at higher levels

![Pruning Comparison](../pruning_results/pruning_strategy_comparison.png)

## Performance Metrics

We evaluate pruning effectiveness through multiple metrics:

### Inference Speed

We measure tokens per second during text generation to quantify the computational speedup from pruning:

```python
def measure_inference_speed(model, tokenizer, prompt, num_tokens=50, num_runs=3):
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.generate(**inputs, max_length=len(inputs.input_ids[0]) + num_tokens)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    tokens_per_second = num_tokens / avg_time
    return tokens_per_second
```

### Text Quality

We use lexical diversity and repetition metrics to ensure pruning doesn't degrade generation quality:

```python
def compute_text_quality_metrics(text):
    words = text.split()
    
    # Lexical diversity (higher is better)
    unique_words = len(set(words))
    total_words = len(words)
    diversity = unique_words / total_words
    
    # Repetition score (lower is better)
    window_size = min(50, len(words))
    repeats = 0
    for i in range(len(words) - 1):
        end_idx = min(i + window_size, len(words))
        if words[i] in words[i+1:end_idx]:
            repeats += 1
            
    repetition_score = repeats / (len(words) - 1)
    
    return {
        "repetition_score": repetition_score,
        "lexical_diversity": diversity
    }
```

### Perplexity

For more rigorous evaluation, we calculate perplexity on validation datasets:

```python
def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Calculate loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()
            
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            total_loss += loss.item()
            total_tokens += (shift_labels != -100).sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity
```

## Conclusion

Our pruning methodology demonstrates that transformer models can operate effectively with significantly fewer attention heads than they're typically built with. By strategically identifying and pruning less important heads, we achieve:

1. **Improved computational efficiency** (faster inference)
2. **Reduced memory requirements** (fewer active parameters)
3. **Maintained model performance** (comparable text quality)

The Sentinel-AI framework's pruning capabilities enable more efficient deployment of transformer models across a variety of hardware environments, particularly resource-constrained devices where computational efficiency is critical.