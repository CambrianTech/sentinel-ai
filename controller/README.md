# Controller Module

This module implements the neural network controller that manages the dynamic pruning and regrowth of attention heads in the Adaptive Transformer.

## Key Components

### `controller_manager.py`
Central controller that coordinates:
- Dynamic pruning based on attention head metrics
- U-Net style skip connections between encoder and decoder layers
- Integration with the model during training and inference

### `controller_ann.py`
Neural network implementation that:
- Processes metrics like entropy and gradient norms
- Makes decisions about which heads to prune or expand
- Learns patterns for optimal architecture configuration

### `/metrics`
Collection of metrics used to evaluate attention head importance:
- Attention entropy
- Gradient norms
- Head utilization metrics

### `/visualizations`
Tools for visualizing controller behavior:
- Gate activity heatmaps
- Pruning patterns over time
- U-Net connection visualization

## Usage

```python
from controller.controller_manager import ControllerManager

# Initialize controller with model
controller = ControllerManager(
    model=model,
    num_layers=12,
    num_heads=12,
    threshold=0.1,
    l1_lambda=0.001
)

# Enable U-Net skip connections
controller.enable_unet_connections(enable=True, connection_scale=0.1)

# Get regularization loss for training
reg_loss = controller.get_regularization_loss()

# Update gate values based on metrics
controller.update_gates(entropy_values, gradient_norms)
```

## Implementation Details

The controller works in phases:
1. **Monitoring Phase**: Collect metrics on attention head activity
2. **Pruning Phase**: Disable underutilized heads by reducing gate values
3. **Regrowth Phase**: Selectively reactivate heads for complex inputs
4. **U-Net Connections**: Skip connections help stabilize pruned/regrown heads

See implementation details in [`/docs/implementation_details.md`](/docs/implementation_details.md).