# Per-Head Learning Rate Adjustment

This document describes the implementation of per-head learning rate adjustments in Sentinel-AI, a feature designed to improve adaptation of attention heads when they are pruned or regrown during training.

## Motivation

When attention heads are dynamically pruned or regrown during training, they need to quickly adapt to their new role in the network. Using the same learning rate for all parameters can lead to suboptimal adaptation:

1. **Newly activated heads** need higher learning rates to quickly learn useful patterns
2. **Recently pruned heads** (that may be reactivated later) benefit from modified learning rates
3. **Stable heads** should maintain lower, stable learning rates to avoid disrupting learned patterns

By implementing per-head learning rate adjustments, we enable more effective "surgery" on the network architecture during training, allowing parts that undergo changes to adapt more quickly.

## Implementation Details

### HeadLRManager Class

The implementation centers around the `HeadLRManager` class in `utils/head_lr_manager.py`:

```python
class HeadLRManager:
    def __init__(
        self,
        model,
        optimizer,
        base_lr,
        boost_factor=5.0,
        decay_factor=0.9,
        warmup_steps=200,
        cooldown_steps=1000
    ):
        # ...
```

Key components include:

1. **Head Status Tracking**: Maintains the status of each attention head:
   - `-1`: Pruned (inactive)
   - `0`: Stable (active, no recent changes)
   - `>0`: Recently activated (value indicates steps since activation)

2. **Learning Rate Adjustment Cycle**:
   - **Warmup Phase**: When a head is newly activated, its learning rate gradually increases from the base rate to `base_lr * boost_factor` over `warmup_steps` steps
   - **Cooldown Phase**: After warmup, the learning rate exponentially decays back to the base rate with factor `decay_factor` over `cooldown_steps - warmup_steps` steps
   - **Stable Phase**: Once cooldown completes, the head returns to the base learning rate

3. **Parameter Group Mapping**: Maps each attention head to its corresponding parameter groups in the optimizer, allowing precise learning rate control

### Integration with Controller

The `ControllerManager` has been updated to work with the `HeadLRManager`:

1. It tracks gate value changes between updates to detect pruning and regrowth events
2. It passes this information to the `HeadLRManager` to update head status and learning rates
3. It includes detailed metrics about learning rate adjustments in its return values

```python
# In ControllerManager.step():
# Get current gate values before update (for tracking changes)
prev_gate_values = self.controller.forward()

# After gate updates:
if head_lr_manager is not None:
    # Update head status based on gate changes
    head_status_info = head_lr_manager.update_head_status(gate_values, prev_gate_values)
    
    # Update learning rates based on status
    lr_update_info = head_lr_manager.update_learning_rates()
```

### Training Integration

The training script has been updated to:

1. Initialize the `HeadLRManager` if enabled
2. Pass it to the controller during update steps
3. Log metrics about learning rate adjustments
4. Save and load head learning rate state during checkpointing

## Configuration Parameters

The feature can be configured through command-line arguments:

```
--enable_head_lr              Enable per-head learning rate adjustments
--head_lr_boost FLOAT         Boost factor for newly activated heads (default: 5.0)
--head_lr_decay FLOAT         Decay factor for head learning rate (default: 0.9)
--head_lr_warmup INT          Warmup steps for head learning rates (default: 200)
--head_lr_cooldown INT        Cooldown steps for head learning rates (default: 1000)
```

## Learning Rate Profiles

With default settings, the learning rate profile for a newly activated head follows this pattern:

1. At activation (step 0): LR = base_lr
2. During warmup (steps 1-200): LR gradually increases to base_lr * 5.0
3. During cooldown (steps 201-1000): LR decays exponentially back to base_lr
4. After cooldown: LR stays at base_lr

This profile allows for fast adaptation immediately after architectural changes while ensuring stability in the long run.

## Metrics and Monitoring

The system tracks several metrics related to per-head learning rates:

- `newly_activated_heads`: Number of heads newly activated in this update
- `newly_pruned_heads`: Number of heads newly pruned in this update
- `cooling_down_heads`: Number of heads currently in the warmup/cooldown phase
- `max_head_lr_multiplier`: Maximum learning rate multiplier currently applied
- `avg_head_lr_multiplier`: Average learning rate multiplier across all heads

These metrics are logged during training to help monitor the adaptation process.

## Future Improvements

Potential enhancements to this feature include:

1. **Adaptive boost factors**: Automatically determine optimal boost factors based on layer position and head importance
2. **Task-specific adaptation**: Modify learning rate profiles based on task characteristics
3. **Gradient-driven adjustments**: Use gradient magnitudes to further refine per-head learning rates
4. **Visualization tools**: Create visualizations of learning rate changes during training