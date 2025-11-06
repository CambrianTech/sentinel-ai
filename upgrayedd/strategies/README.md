# Upgrayedd Pruning Strategies

This module contains the pruning and growth strategies for the Upgrayedd model optimization system. These strategies determine which attention heads to prune and where to regrow capacity during the neural plasticity cycle.

## Available Strategies

### RandomPruningStrategy

A baseline strategy that randomly selects heads to prune and grow without considering importance or performance metrics. Useful as a control to compare against more sophisticated strategies.

```python
from upgrayedd.strategies import RandomPruningStrategy

strategy = RandomPruningStrategy(
    pruning_ratio=0.3,  # Prune 30% of heads
    growth_ratio=0.1,   # Regrow 10% of pruned heads
    min_heads=1,        # Keep at least 1 head per layer
    seed=42             # Random seed for reproducibility
)
```

### EntropyPruningStrategy

Prunes heads based on attention entropy, targeting heads with higher entropy (more uniform attention patterns) as they are likely less specialized. For regrowth, it prioritizes layers with low average entropy in remaining heads, as these layers are performing more specialized functions.

```python
from upgrayedd.strategies import EntropyPruningStrategy

strategy = EntropyPruningStrategy(
    pruning_ratio=0.3,
    growth_ratio=0.1,
    min_heads=1,
    seed=42,
    attention_samples=128,     # Number of samples to collect attention patterns
    use_cached_entropy=True    # Reuse entropy values if available
)
```

### MagnitudePruningStrategy

Prunes heads based on the magnitude (L2 norm) of their weights, assuming that heads with smaller weight norms contribute less to the model's performance. For regrowth, it prioritizes layers with high average magnitude in remaining heads, which are actively learning important features.

```python
from upgrayedd.strategies import MagnitudePruningStrategy

strategy = MagnitudePruningStrategy(
    pruning_ratio=0.3,
    growth_ratio=0.1,
    min_heads=1,
    seed=42,
    weight_threshold=0.01,     # Threshold below which heads are considered unimportant
    use_cached_magnitudes=True # Reuse magnitude values if available
)
```

## Using Strategies

You can use the `get_strategy` factory function to create a strategy by name:

```python
from upgrayedd.strategies import get_strategy

strategy = get_strategy(
    "entropy",  # Strategy name: "random", "entropy", or "magnitude"
    pruning_ratio=0.3,
    growth_ratio=0.1
)
```

## Custom Strategies

To create a custom pruning strategy, extend the `BasePruningStrategy` class and implement the required methods:

```python
from upgrayedd.strategies.base import BasePruningStrategy

class MyCustomStrategy(BasePruningStrategy):
    def __init__(self, pruning_ratio=0.3, growth_ratio=0.1, min_heads=1, seed=None):
        super().__init__(pruning_ratio, growth_ratio, min_heads, seed)
        # Custom initialization
        
    def select_heads_to_prune(self, model, head_importances):
        # Your custom logic to select heads for pruning
        # Returns: Dict[str, List[int]] mapping layer names to head indices
        pass
        
    def select_heads_to_grow(self, model, pruned_heads, metrics):
        # Your custom logic to select layers for head growth
        # Returns: Dict[str, int] mapping layer names to number of heads to grow
        pass
```

## Extending the Framework

The strategy system is designed to be extensible. You can add your own strategies to target specific model behaviors or optimize for particular objectives.

Some ideas for additional strategies:

- **Attention-based**: Prune heads that attend to similar positions
- **Loss-impact**: Prune heads with least impact on validation loss
- **Gradient-based**: Prune heads with consistently small gradients
- **Task-specific**: Prune heads less important for specific downstream tasks

## Integration with Upgrayedd Pipeline

Strategies are integrated into the Upgrayedd pipeline:

```python
from upgrayedd import UpgrayeddPipeline
from upgrayedd.strategies import EntropyPruningStrategy

# Create a strategy
strategy = EntropyPruningStrategy(pruning_ratio=0.3, growth_ratio=0.1)

# Initialize the pipeline with the strategy
pipeline = UpgrayeddPipeline(
    model=model,
    tokenizer=tokenizer,
    strategy=strategy,
    device="cuda"
)

# Run optimization
optimized_model = pipeline.run_optimization(
    dataloader=dataloader,
    num_cycles=3,
    epochs_per_cycle=1
)
```