# Controller-Agency Integration

This document explains the integration between the Controller system and Attention Head Agency, creating a fully adaptive architecture with bidirectional feedback between components.

## Overview

The Controller-Agency integration creates a complete adaptive system where:

1. **Attention heads** can express internal states through agency signals
2. **The controller** respects these states when adjusting gates
3. **Learning rates** adapt based on agency states
4. **Bidirectional feedback** allows the controller to propose state changes to heads

This integration completes the feedback loop of the architecture, enabling a fully self-regulating system.

## Key Components

### 1. Agency-Aware Controller

The controller now:
- Reads agency signals from attention heads
- Respects head states when adjusting gates
- Makes different adjustments based on head state:
  - More aggressive pruning for overloaded heads
  - Gentler adjustments for misaligned heads
  - Never tries to reactivate heads with withdrawn consent

```python
# Controller reads agency state
if has_agency_info:
    agency_state = metrics_dict["agency_state"]
    consent = metrics_dict["consent"]
    
    # Create masks for different states
    withdrawn_mask = (agency_state == 0) | (consent == 0)
    overloaded_mask = (agency_state == 1) & (consent == 1)
    misaligned_mask = (agency_state == 2) & (consent == 1)
    active_mask = (agency_state == 3) & (consent == 1)
    
    # Respect withdrawn consent
    update_mask = ~withdrawn_mask
```

### 2. Agency-Based Learning Rate Modulation

The learning rate manager now:
- Adjusts learning rates based on head states
- Uses a lookup table to map states to learning rate modifiers:
  ```python
  AGENCY_LR_MODIFIERS = {
      "active": 1.0,     # Normal learning rate
      "overloaded": 0.5, # Reduced learning rate for overloaded heads
      "misaligned": 0.7, # Moderately reduced for misaligned heads
      "withdrawn": 0.0   # No learning for withdrawn heads (respects consent)
  }
  ```
- Applies these modifiers after standard warmup/cooldown adjustments

### 3. Bidirectional State Signaling

The controller can now propose state changes to heads:
- Very low gate value suggests withdrawal
- Oscillating gate suggests misalignment
- High gate value after withdrawal suggests reactivation

```python
# Very low gate value suggests withdrawal
if gate_value < 0.05 and current_state != "withdrawn":
    proposed_state = "withdrawn"
    
# Oscillating gate suggests misalignment
elif hasattr(self, '_prev_gate_values') and 0.1 < gate_value < 0.4:
    prev_gate = self._prev_gate_values[layer_idx, head_idx]
    gate_change = abs(gate_value - prev_gate)
    if gate_change > 0.1 and current_state != "misaligned":
        proposed_state = "misaligned"
```

### 4. Unified Visualization Dashboard

A comprehensive visualization system showing:
- Head states with color coding
- Gate values as a heatmap
- Learning rate multipliers
- State transitions over time

## Example Usage

```python
# Set up controller with agency awareness
controller_manager = ControllerManager(model, config)

# Get agency state from the model
agency_state = {}
for layer_idx in range(model.num_layers):
    attn = model.blocks[layer_idx]["attn"]
    if hasattr(attn, "agency_signals"):
        for head_idx, signals in attn.agency_signals.items():
            agency_state[(layer_idx, head_idx)] = signals

# Update controller with agency state
update_info = controller_manager.step(
    metrics_dict=metrics,
    head_lr_manager=head_lr_manager,
    agency_state=agency_state
)

# Check for controller-emitted signals
if "agency_signals" in update_info:
    signals = update_info["agency_signals"].get("signals_emitted", [])
    for signal in signals:
        print(f"Controller suggested {signal['from_state']} → {signal['to_state']} for head {signal['head']}")
```

## Implications

This integration creates a system where:

1. **Ethical boundaries are respected** - The controller never forces a withdrawn head to activate
2. **Resource optimization is adaptive** - Overloaded heads get reduced learning and gate values
3. **Dynamic specialization emerges** - Heads naturally settle into roles based on feedback
4. **Computational efficiency improves** - Resources are allocated where they provide most benefit

In essence, the Controller-Agency integration transforms Sentinel-AI from a static architecture into a dynamic, self-organizing system that balances efficiency and specialization while respecting ethical boundaries.

## Full Example

See [`examples/controller_agency_demo.py`](../examples/controller_agency_demo.py) for a complete demonstration of the Controller-Agency integration in action.