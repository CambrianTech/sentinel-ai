# Neural Plasticity System

The Neural Plasticity system provides dynamic pruning and recovery of attention heads in transformer models, using entropy and gradient-based metrics to make intelligent decisions about model structure during training.

## Architecture

The system is organized into the following components:

1. **Core Module Structure**:
   - `sentinel.pruning.plasticity_controller`: Main controller for the neural plasticity system
   - `sentinel.pruning.dual_mode_pruning`: Implementation of the pruning mechanics and modes
   - `sentinel.plasticity`: Higher-level modules for plasticity experiments

2. **Key Classes**:
   - `PlasticityController`: Makes pruning decisions based on attention metrics
   - `PruningMode`: Enumeration of pruning modes (ADAPTIVE, COMPRESSED)
   - `NeuralPlasticityExperiment`: Orchestrates complete experiments 

## Running Experiments

### Quick Start

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run a minimal experiment with synthetic data
python scripts/run_neural_plasticity.py --quick_test
```

### Complete Experiment

```bash
# Run with real data
python scripts/run_neural_plasticity.py \
  --model_name distilgpt2 \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --pruning_strategy entropy \
  --pruning_level 0.2 \
  --pruning_mode adaptive
```

### Output Files

All experiment output is stored in the consolidated `experiment_output/neural_plasticity/` directory:

```
experiment_output/
  neural_plasticity/
    run_YYYYMMDD_HHMMSS/    # Timestamp-based run directory
      warmup/               # Warmup phase data
      pruning/              # Pruning decisions
      fine_tuning/          # Fine-tuning results
      metrics/              # Performance metrics
      visualizations/       # Visualizations and plots
      dashboards/           # HTML dashboards
      models/               # Saved model checkpoints
      experiment.log        # Complete experiment log
```

### Modes

- **Adaptive Mode**: Allows both pruning and recovery of attention heads
- **Compressed Mode**: Only allows pruning for permanent model compression

## Test Suite

The neural plasticity system includes comprehensive tests:

```bash
# Run unit tests
pytest tests/unit/sentinel/pruning/test_plasticity_controller.py

# Run integration tests
pytest tests/sentinel/pruning/integration_test_plasticity.py
```

## Implementation Details

### Metrics for Pruning Decisions

The system uses two primary metrics to make pruning decisions:

1. **Attention Entropy**: Measures how diffuse or focused a head's attention pattern is
   - Higher entropy: More uniform attention distribution (less focused)
   - Lower entropy: More peaked attention distribution (more focused)

2. **Gradient Magnitude**: Measures how much a head is learning
   - Higher gradient: Head is actively learning and important
   - Lower gradient: Head is not changing much during training

### Decision Logic

The plasticity controller uses the following logic:

1. **Pruning** (zeroing) a head when:
   - Entropy is high (unfocused attention)
   - Gradient is low (not learning much)

2. **Reviving** a previously pruned head when:
   - Entropy is low (focused attention)
   - Gradient is high (would learn a lot if activated)
   - Has been pruned for minimum required epochs

3. **Keeping** a head in its current state otherwise

This allows the model to dynamically adapt its structure based on actual usage patterns during training or fine-tuning.

## Dashboard and Visualization

The system includes visualization features to monitor neural plasticity:

1. **Head Entropy Visualization**: Heat maps of attention entropy
2. **Gradient Visualization**: Heat maps of gradient norms
3. **Pruning History**: Tracking of pruning and revival decisions
4. **Sparsity Metrics**: Monitoring overall model sparsity

## References

- [Attention is all you need](https://arxiv.org/abs/1706.03762)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
- [Pruning Attention Heads for Transfer Learning](https://arxiv.org/abs/1905.09418)