# Pruning Stress Evaluation

This document describes how to use the pruning stress evaluation system to measure how pruned models adapt to challenging tasks and recover from stress.

## 👾 Overview

The Pruning Stress Evaluation framework combines Sentinel-AI's pruning capabilities with the stress test protocols to investigate how different pruning strategies affect a model's neural plasticity. This integration allows for rigorous testing of:

1. **Recovery Capacity**: How well pruned models recover performance after task switches
2. **Forgetting Resistance**: How well pruned models maintain performance on previous tasks
3. **Adaptation Efficiency**: How quickly pruned models adapt to new tasks after pruning

## Components

The framework integrates several Sentinel-AI modules:

- **Pruning Module**: For creating pruned model variants using different strategies
- **Stress Protocol System**: For testing adaptation through task alternation
- **Entropy Journal**: For tracking attention entropy patterns during recovery
- **Function Tracker**: For measuring function preservation during adaptation

## Usage Examples

### Basic Usage

```bash
python scripts/pruning_stress_evaluation.py \
    --model distilgpt2 \
    --strategies entropy,random,magnitude \
    --levels 0.1,0.3,0.5 \
    --protocol diverse \
    --cycles 3 \
    --epochs 1 \
    --output_dir outputs/pruning_stress_results
```

This will:
1. Load the specified model (distilgpt2)
2. Create pruned variants using 3 strategies × 3 pruning levels
3. Run the diverse task alternation protocol on each variant
4. Track entropy and function preservation during adaptation
5. Generate comparative visualizations across strategies and levels

### More Examples

**Memory Stress Testing**:
```bash
python scripts/pruning_stress_evaluation.py \
    --model facebook/opt-125m \
    --strategies entropy,random \
    --levels 0.1,0.3 \
    --protocol memory \
    --output_dir outputs/memory_stress_results
```

**Conflicting Tasks Test**:
```bash
python scripts/pruning_stress_evaluation.py \
    --model EleutherAI/pythia-70m \
    --strategies entropy \
    --levels 0.1,0.3,0.5,0.7 \
    --protocol conflict \
    --cycles 5 \
    --output_dir outputs/conflict_recovery
```

**Disable Tracking**:
```bash
python scripts/pruning_stress_evaluation.py \
    --model distilgpt2 \
    --no_entropy \
    --no_function
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name or path | `distilgpt2` |
| `--strategies` | Comma-separated list of pruning strategies | `entropy,random` |
| `--levels` | Comma-separated list of pruning levels | `0.1,0.3,0.5` |
| `--protocol` | Stress protocol to use (`diverse`, `memory`, `conflict`) | `diverse` |
| `--cycles` | Number of task alternation cycles | `3` |
| `--epochs` | Epochs per task | `1` |
| `--output_dir` | Directory to save results | `outputs/pruning_stress_eval` |
| `--device` | Device to run on | auto-detect |
| `--seed` | Random seed | `42` |
| `--no_entropy` | Disable entropy tracking | `False` |
| `--no_function` | Disable function preservation tracking | `False` |

## Output Structure

The evaluation generates a structured output directory:

```
outputs/pruning_stress_eval/
├── pruning_stress_results.json       # All experiment results
├── experiment_summaries.json         # Summary metrics across experiments
├── diverse_comparison/               # Comparative visualizations
│   ├── recovery_rate_comparison.png
│   ├── forgetting_rate_comparison.png
│   └── performance_comparison.png
├── diverse_entropy_0.3/              # Experiment for entropy at 0.3 level
│   ├── entropy_tracking/             # Entropy journal outputs
│   │   ├── visualizations/
│   │   ├── entropy_journal.jsonl
│   │   └── entropy_journal_summary.md
│   ├── function_tracking/            # Function preservation outputs
│   ├── metrics_history.json          # Protocol metrics
│   └── visualizations/               # Protocol visualizations
└── ...                               # Other experiment directories
```

## Metrics Tracked

The framework tracks several key metrics:

1. **Recovery Rate**: How quickly a model recovers performance on a task after returning to it
2. **Forgetting Rate**: How much performance is lost on previous tasks after learning new ones
3. **Adaptation Efficiency**: How many epochs are needed to reach target performance on new tasks
4. **Entropy Patterns**: How attention entropy evolves during recovery and adaptation
5. **Function Preservation**: How well the original model's function is preserved after pruning and adaptation

## Visualizations

The system generates several types of visualizations:

1. **Heatmaps**: Comparing metrics across pruning strategies and levels
2. **Radar Charts**: Multi-dimensional comparison of pruning strategies
3. **Time Series**: Showing performance, entropy, and function preservation over time
4. **EEG-like Plots**: Showing entropy patterns during adaptation (if entropy tracking is enabled)

## Integration with Other Scripts

The pruning stress evaluation system can be integrated with other Sentinel-AI scripts:

```python
from scripts.pruning_stress_evaluation import PruningStressEvaluator

# Create evaluator
evaluator = PruningStressEvaluator(
    model_name="my_custom_model",
    pruning_strategies=["entropy", "random"],
    pruning_levels=[0.1, 0.3, 0.5],
    output_dir="outputs/custom_eval"
)

# Run evaluation
results = evaluator.run_evaluation(
    protocol_type="diverse",
    cycles=3,
    epochs_per_task=1
)

# Access results
for strategy, level in results["pruning_results"].items():
    print(f"Strategy {strategy}: Recovery rate = {level['avg_recovery_rate']:.2f}")
```

## Research Applications

This framework is designed to answer research questions like:

1. Which pruning strategies preserve the most neural plasticity?
2. How does pruning level affect adaptation and recovery capabilities?
3. Is there a trade-off between model size and adaptability?
4. Do certain architectures maintain better plasticity after pruning?
5. Can pruning actually improve adaptation to certain types of tasks?

## Next Steps

Future enhancements to the pruning stress evaluation system could include:

1. Integration with the RL controller for auto-optimizing pruning for plasticity
2. Adding more diverse and challenging task suites
3. Supporting comparative analysis across model families
4. Longitudinal tracking across multiple pruning and recovery cycles
5. Adaptive difficulty adjustment based on model performance