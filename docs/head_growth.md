# Head Growth Implementation

This document details how Sentinel-AI implements attention head growth after pruning, completing the neural plasticity cycle.

## Overview

The head growth mechanism allows a model to strategically add new attention heads after pruning, targeting areas where additional capacity is most beneficial. This process follows these key principles:

1. **Gradual Integration** - New heads are added with minimal initial influence to prevent disruption
2. **Strategic Placement** - Heads are added in layers and positions where they'll have the most impact
3. **Knowledge Transfer** - U-Net skip connections help new heads learn from related patterns
4. **Specialized Learning** - New heads receive higher learning rates during initial training

## Implementation

The core implementation is in the `grow_attention_heads_gradually()` function:

```python
def grow_attention_heads_gradually(pruning_module, growth_percentage=0.05, strategy="gradient_sensitivity", 
                                 initial_scale=0.01, warmup_steps=100):
    """
    Gradually grow new attention heads to prevent performance collapse.
    
    Args:
        pruning_module: The pruning module containing the model
        growth_percentage: Percentage of new heads to add
        strategy: Strategy to determine where to add heads
        initial_scale: Initial scaling factor for new head weights (small to start)
        warmup_steps: Number of steps to linearly increase head influence
        
    Returns:
        new_params: Model parameters with new heads added
        added_count: Number of heads added
        added_heads: List of (layer, head) tuples where heads were added
        warmup_schedule: Function to update head scaling during warmup
    """
    # Get model parameters and determine current active heads
    params = pruning_module.get_params()
    active_heads = pruning_module.get_active_heads()
    total_heads = pruning_module.get_total_head_count()
    
    # Determine number of heads to add
    inactive_heads = total_heads - len(active_heads)
    heads_to_add = max(1, int(growth_percentage * total_heads))
    heads_to_add = min(heads_to_add, inactive_heads)  # Can't add more than inactive count
    
    if heads_to_add == 0:
        return params, 0, [], lambda step: 1.0
    
    # Select which heads to add based on strategy
    head_candidates = []
    
    if strategy == "gradient_sensitivity":
        # Get gradient sensitivity scores for all possible head positions
        sensitivity_scores = calculate_gradient_sensitivity(pruning_module)
        # Sort inactive heads by sensitivity
        for layer_idx, head_idx, score in sensitivity_scores:
            if (layer_idx, head_idx) not in active_heads:
                head_candidates.append((layer_idx, head_idx, score))
        
        # Sort by sensitivity score (highest first)
        head_candidates.sort(key=lambda x: x[2], reverse=True)
        
    elif strategy == "entropy_gap":
        # Identify areas with high attention entropy gap
        entropy_gaps = calculate_entropy_gaps(pruning_module)
        for layer_idx, head_idx, gap in entropy_gaps:
            if (layer_idx, head_idx) not in active_heads:
                head_candidates.append((layer_idx, head_idx, gap))
                
        # Sort by entropy gap (highest first)
        head_candidates.sort(key=lambda x: x[2], reverse=True)
        
    elif strategy == "balanced":
        # Ensure balanced distribution of heads across layers
        layer_counts = defaultdict(int)
        for layer_idx, _ in active_heads:
            layer_counts[layer_idx] += 1
            
        # Identify layers with fewest active heads
        layer_ranking = [(layer_idx, count) for layer_idx, count in layer_counts.items()]
        layer_ranking.sort(key=lambda x: x[1])
        
        # Add heads to layers with fewest active heads
        for layer_idx, _ in layer_ranking:
            for head_idx in range(pruning_module.num_heads_per_layer):
                if (layer_idx, head_idx) not in active_heads:
                    head_candidates.append((layer_idx, head_idx, 1.0))
    
    else:  # Default to random selection
        # Get all inactive heads
        for layer_idx in range(pruning_module.num_layers):
            for head_idx in range(pruning_module.num_heads_per_layer):
                if (layer_idx, head_idx) not in active_heads:
                    head_candidates.append((layer_idx, head_idx, random.random()))
                    
        # Shuffle randomly
        random.shuffle(head_candidates)
    
    # Select top candidates based on heads_to_add
    selected_heads = [(layer_idx, head_idx) for layer_idx, head_idx, _ in head_candidates[:heads_to_add]]
    
    # Initialize new heads with small weights
    new_params = copy.deepcopy(params)
    
    for layer_idx, head_idx in selected_heads:
        # Initialize small weights for Q, K, V projections
        new_params[f"transformer.h.{layer_idx}.attn.q_proj.weight"][:, head_idx*head_dim:(head_idx+1)*head_dim] *= initial_scale
        new_params[f"transformer.h.{layer_idx}.attn.k_proj.weight"][:, head_idx*head_dim:(head_idx+1)*head_dim] *= initial_scale
        new_params[f"transformer.h.{layer_idx}.attn.v_proj.weight"][:, head_idx*head_dim:(head_idx+1)*head_dim] *= initial_scale
        
        # Set gate value to small positive value
        new_params[f"transformer.h.{layer_idx}.attn.gate"][head_idx] = initial_scale
    
    # Create warmup schedule function for gradually increasing head influence
    def warmup_schedule(step):
        if step >= warmup_steps:
            return 1.0
        return min(1.0, initial_scale + (1.0 - initial_scale) * (step / warmup_steps))
    
    return new_params, len(selected_heads), selected_heads, warmup_schedule
```

## Growth Strategies

Sentinel-AI supports multiple strategies for deciding where to add new heads:

### 1. Gradient Sensitivity

The gradient sensitivity strategy identifies locations where adding a head would have the most impact on the loss function:

```python
def calculate_gradient_sensitivity(pruning_module):
    """
    Calculate gradient-based sensitivity scores for possible head positions.
    Higher scores indicate positions where adding a head would have more impact.
    """
    sensitivity_scores = []
    
    # Get a small batch of data
    eval_batch = pruning_module.get_eval_batch()
    
    # For each possible head position
    for layer_idx in range(pruning_module.num_layers):
        for head_idx in range(pruning_module.num_heads_per_layer):
            # Skip already active heads
            if (layer_idx, head_idx) in pruning_module.get_active_heads():
                continue
                
            # Temporarily add this head with minimal contribution
            with temporarily_add_head(pruning_module, layer_idx, head_idx, scale=0.01):
                # Forward pass with gradient tracking
                outputs = pruning_module.model(eval_batch["input_ids"], 
                                             attention_mask=eval_batch["attention_mask"])
                loss = outputs.loss
                
                # Backward pass to get gradients
                loss.backward()
                
                # Measure gradient magnitude for this head's parameters
                q_grad = pruning_module.model.transformer.h[layer_idx].attn.q_proj.weight.grad[:, head_idx*head_dim:(head_idx+1)*head_dim]
                k_grad = pruning_module.model.transformer.h[layer_idx].attn.k_proj.weight.grad[:, head_idx*head_dim:(head_idx+1)*head_dim]
                v_grad = pruning_module.model.transformer.h[layer_idx].attn.v_proj.weight.grad[:, head_idx*head_dim:(head_idx+1)*head_dim]
                
                # Calculate sensitivity score (norm of gradients)
                sensitivity = torch.norm(q_grad).item() + torch.norm(k_grad).item() + torch.norm(v_grad).item()
                
                sensitivity_scores.append((layer_idx, head_idx, sensitivity))
                
                # Clear gradients
                pruning_module.model.zero_grad()
    
    return sensitivity_scores
```

### 2. Entropy Gap

The entropy gap strategy adds heads where there's a significant difference between the entropy of the current attention pattern and the ideal distribution:

```python
def calculate_entropy_gaps(pruning_module):
    """
    Calculate entropy gaps for possible head positions.
    Higher gaps indicate positions where adding a head could provide more diverse attention.
    """
    entropy_gaps = []
    
    # Get a small batch of data
    eval_batch = pruning_module.get_eval_batch()
    
    # Calculate current attention entropies
    current_entropies = {}
    
    # Forward pass to get attention patterns
    with torch.no_grad():
        outputs = pruning_module.model(eval_batch["input_ids"], 
                                     attention_mask=eval_batch["attention_mask"],
                                     output_attentions=True)
        
        attentions = outputs.attentions
        
        # Calculate entropy for each layer
        for layer_idx, layer_attention in enumerate(attentions):
            layer_entropy = calculate_attention_entropy(layer_attention)
            current_entropies[layer_idx] = layer_entropy
    
    # For each possible head position
    for layer_idx in range(pruning_module.num_layers):
        # Skip layers with all heads active
        active_in_layer = sum(1 for l, h in pruning_module.get_active_heads() if l == layer_idx)
        if active_in_layer >= pruning_module.num_heads_per_layer:
            continue
            
        # Calculate theoretical optimal entropy if we added one head
        optimal_entropy = calculate_optimal_entropy(pruning_module.num_heads_per_layer)
        
        # Calculate entropy gap
        current = current_entropies.get(layer_idx, 0)
        gap = optimal_entropy - current
        
        # For each inactive head in this layer
        for head_idx in range(pruning_module.num_heads_per_layer):
            if (layer_idx, head_idx) not in pruning_module.get_active_heads():
                entropy_gaps.append((layer_idx, head_idx, gap))
    
    return entropy_gaps
```

### 3. Balanced Distribution

The balanced strategy ensures heads are distributed evenly across layers:

```python
def get_balanced_head_candidates(pruning_module):
    """
    Select head candidates to ensure balanced distribution across layers.
    """
    # Count active heads per layer
    layer_counts = defaultdict(int)
    for layer_idx, _ in pruning_module.get_active_heads():
        layer_counts[layer_idx] += 1
        
    # Identify layers with fewest active heads
    layer_ranking = [(layer_idx, count) for layer_idx, count in layer_counts.items()]
    layer_ranking.sort(key=lambda x: x[1])
    
    candidates = []
    # Add heads to layers with fewest active heads
    for layer_idx, _ in layer_ranking:
        for head_idx in range(pruning_module.num_heads_per_layer):
            if (layer_idx, head_idx) not in pruning_module.get_active_heads():
                candidates.append((layer_idx, head_idx))
    
    return candidates
```

## Integration with Differential Learning Rates

New heads benefit from specialized learning rates during initial training:

```python
def setup_differential_learning_rates(model, optimizer, added_heads, lr_multiplier=5.0):
    """
    Configure higher learning rates for newly added attention heads.
    
    Args:
        model: The model with newly added heads
        optimizer: The optimizer
        added_heads: List of (layer_idx, head_idx) tuples for new heads
        lr_multiplier: Factor by which to increase learning rate for new heads
    """
    head_params = {}
    
    # Identify parameters for new heads
    for layer_idx, head_idx in added_heads:
        # Get parameter tensors for this head
        q_params = model.transformer.h[layer_idx].attn.q_proj.weight[:, head_idx*head_dim:(head_idx+1)*head_dim]
        k_params = model.transformer.h[layer_idx].attn.k_proj.weight[:, head_idx*head_dim:(head_idx+1)*head_dim]
        v_params = model.transformer.h[layer_idx].attn.v_proj.weight[:, head_idx*head_dim:(head_idx+1)*head_dim]
        gate_param = model.transformer.h[layer_idx].attn.gate[head_idx]
        
        # Add to dictionary with parameter handle as key
        head_params[id(q_params)] = lr_multiplier
        head_params[id(k_params)] = lr_multiplier
        head_params[id(v_params)] = lr_multiplier
        head_params[id(gate_param)] = lr_multiplier
    
    # Set learning rate multipliers in optimizer
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param_id = id(param)
            if param_id in head_params:
                param_group['lr_multiplier'] = head_params[param_id]
            else:
                param_group['lr_multiplier'] = 1.0
```

## U-Net Skip Connections for Knowledge Transfer

The U-Net skip connections enable knowledge transfer to new heads:

```python
class UNetSkipConnection(nn.Module):
    """
    Implements a U-Net-style skip connection to transfer knowledge between layers.
    """
    def __init__(self, embed_dim, source_layer_idx, target_layer_idx):
        super().__init__()
        self.source_layer_idx = source_layer_idx
        self.target_layer_idx = target_layer_idx
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, source_features, target_features):
        """
        Fuse features from source and target layers.
        
        Args:
            source_features: Hidden states from the source layer
            target_features: Hidden states from the target layer
            
        Returns:
            Fused features
        """
        # Concatenate features
        combined = torch.cat([source_features, target_features], dim=-1)
        
        # Fuse using linear projection
        fused = self.fusion(combined)
        
        return fused
```

The AdaptiveTransformer ensures these skip connections are used when initializing new heads:

```python
def _init_new_head_with_skip_connection(self, layer_idx, head_idx):
    """
    Initialize a new attention head using knowledge from earlier layers via skip connections.
    """
    # Find a suitable source layer (typically earlier in the network)
    source_layer_idx = max(0, layer_idx - 2)
    
    # Get source layer hidden states
    source_hidden_states = self.saved_hidden_states[source_layer_idx]
    
    # Get current layer hidden states
    target_hidden_states = self.hidden_states[layer_idx]
    
    # Apply skip connection fusion
    if not hasattr(self, f"skip_fusion_{source_layer_idx}_{layer_idx}"):
        self.add_module(
            f"skip_fusion_{source_layer_idx}_{layer_idx}", 
            UNetSkipConnection(self.embed_dim, source_layer_idx, layer_idx)
        )
    
    skip_fusion = getattr(self, f"skip_fusion_{source_layer_idx}_{layer_idx}")
    fused_knowledge = skip_fusion(source_hidden_states, target_hidden_states)
    
    # Use this fused knowledge to initialize the new head's key and value projections
    with torch.no_grad():
        # Extract patterns from fused knowledge
        patterns = self._extract_attention_patterns(fused_knowledge)
        
        # Use these patterns to initialize the head's parameters
        self._apply_patterns_to_head(patterns, layer_idx, head_idx)
```

## Warmup Schedule for New Heads

New heads are gradually integrated using a warmup schedule:

```python
class GradualHeadWarmupScheduler:
    """
    Scheduler that gradually increases the influence of newly added heads.
    """
    def __init__(self, model, added_heads, initial_scale=0.01, warmup_steps=100):
        self.model = model
        self.added_heads = added_heads
        self.initial_scale = initial_scale
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def step(self):
        """
        Update the scaling factor for new heads based on current step.
        """
        self.current_step += 1
        
        # Calculate current scale
        if self.current_step >= self.warmup_steps:
            scale = 1.0
        else:
            # Linear warmup
            scale = self.initial_scale + (1.0 - self.initial_scale) * (self.current_step / self.warmup_steps)
        
        # Apply scaling to gate values
        with torch.no_grad():
            for layer_idx, head_idx in self.added_heads:
                # Get original gate value (initialized to 1.0)
                original_gate = 1.0
                
                # Scale the gate
                self.model.transformer.h[layer_idx].attn.gate[head_idx] = original_gate * scale
```

## Usage Example

Here's how to use the head growth functionality in practice:

```python
# Load a pruned model
pruned_model = load_model_from_checkpoint("pruned_model.pth")
pruning_module = PruningModule(pruned_model)

# Grow new heads gradually
new_params, added_count, added_heads, warmup_schedule = grow_attention_heads_gradually(
    pruning_module,
    growth_percentage=0.1,  # Add 10% new heads
    strategy="gradient_sensitivity",
    initial_scale=0.01,  # Start with minimal contribution
    warmup_steps=100      # Gradually increase over 100 steps
)

# Apply new parameters to the model
pruning_module.set_params(new_params)

# Create optimizer with differential learning rates for new heads
optimizer = torch.optim.AdamW(pruned_model.parameters(), lr=5e-5)
setup_differential_learning_rates(pruned_model, optimizer, added_heads, lr_multiplier=5.0)

# Create warmup scheduler
warmup_scheduler = GradualHeadWarmupScheduler(
    pruned_model, added_heads, initial_scale=0.01, warmup_steps=100
)

# Training loop with gradual warmup
for step, batch in enumerate(dataloader):
    # Forward pass
    outputs = pruned_model(**batch)
    loss = outputs.loss
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Update warmup scaling
    warmup_scheduler.step()
    
    # Log training progress
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

## Conclusion

The head growth mechanism completes Sentinel-AI's neural plasticity cycle by allowing models to:

1. Start with small, efficient architectures (pruning)
2. Identify performance gaps (measuring)
3. Add new capacity strategically (growing)
4. Fine-tune the enhanced model (learning)

This approach enables truly dynamic architectures that adapt to changing requirements and optimize themselves for both efficiency and performance.