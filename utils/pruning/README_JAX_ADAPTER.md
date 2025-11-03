# Neural Plasticity JAX Adapter

This directory contains an adapter layer that enables the original JAX-based neural plasticity scripts to work with the newer PyTorch-based implementation in the sentinel-ai codebase.

## Key Files

- `fixed_pruning_module_jax.py`: Main adapter that wraps the PyTorch-based `FixedPruningModule` to provide JAX-compatible interfaces
- `fixed_pruning_module.py`: The PyTorch implementation of the pruning module

## Functionality

The adapter allows the following scripts to continue working:

1. `scripts/neural_plasticity_cycle.py`: Demonstrates the full neural plasticity cycle (train → prune → measure → grow → learn)
2. `scripts/optimize_model_plasticity.py`: Runs multiple neural plasticity cycles to optimize a model for performance, size, and efficiency

## Example Usage

Run the neural plasticity cycle script:
```bash
python scripts/neural_plasticity_cycle.py --model_name distilgpt2 --dataset tiny_shakespeare --cycles 1 --initial_training_steps 100 --learning_steps 100 --eval_samples 5 --eval_every 20
```

Run the model optimization script:
```bash
python scripts/optimize_model_plasticity.py --model_name distilgpt2 --dataset tiny_shakespeare --initial_training_steps 100 --cycle_training_steps 100 --eval_samples 5 --eval_every 20 --max_cycles 1
```

## Implementation Details

The adapter provides compatibility by:

1. Creating a JAX-compatible parameter structure that mirrors the PyTorch model's structure
2. Forwarding key operations to the PyTorch implementation, including:
   - Loading models
   - Pruning attention heads
   - Evaluating perplexity
   - Generating text

## Known Issues

- Perplexity calculation may not be fully accurate due to differences in how JAX and PyTorch handle tensors
- Some advanced features might not work perfectly due to the complexity of bridging between frameworks

## Future Improvements

- Improve perplexity calculation to remove "tuple indices" errors
- Add better error handling for edge cases
- Add unit tests to verify compatibility
- Create a more direct PyTorch implementation that doesn't rely on JAX compatibility