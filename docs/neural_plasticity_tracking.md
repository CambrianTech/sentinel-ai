# Neural Plasticity Tracking System

This document describes the Neural Plasticity Tracking System, a suite of tools for studying how transformer models adapt and reorganize over time. This system allows researchers to observe and analyze the "temporal architecture of intelligence" in neural networks.

## Overview

The Neural Plasticity Tracking System is a scientific instrument for studying how transformer models adapt, learn, and reorganize over time. It treats models as living systems with lifecycles rather than static optimization targets, enabling the study of emergent properties and adaptation patterns.

The system consists of four core modules:
1. **Entropy Journal**: Tracks attention entropy patterns over time
2. **Function Tracking**: Measures functional preservation despite structural changes
3. **Stress Protocols**: Tests model resilience under varying conditions
4. **Entropy Rhythm Plots**: Creates EEG-like visualizations of model adaptation

## Modules

### 1. Entropy Journal (`entropy_journal.py`)

Tracks attention entropy per head over time throughout the neural plasticity cycle. This module enables the study of how entropy patterns evolve as the model learns, adapts, and reorganizes.

Key features:
- Records attention entropy and gate values for each head at each cycle
- Generates visualizations showing entropy patterns and their evolution
- Calculates statistics about entropy changes and head specialization
- Creates summary reports for scientific analysis

Example usage:
```python
from sentinel.plasticity.entropy_journal import EntropyJournal, EntropyJournalConfig

# Create configuration
config = EntropyJournalConfig(
    output_dir="./output/entropy_journal",
    experiment_name="my_experiment",
    create_visualizations=True
)

# Initialize journal
journal = EntropyJournal(config)

# Record model state at different cycles
journal.record_model_state(model, dataloader, cycle_idx=0, cycle_name="Initial")
# ... (after model changes)
journal.record_model_state(model, dataloader, cycle_idx=1, cycle_name="After_Changes")

# Create summary visualizations
journal.visualize_entropy_evolution()
journal.visualize_gate_evolution()
journal.create_summary_report()
```

### 2. Function Tracking (`function_tracking.py`)

Measures how well function is preserved despite structural changes to the model. This module helps understand whether a model maintains its capabilities even as its internal organization evolves.

Key features:
- Compares model behavior before and after changes
- Tracks similarity in internal representations and outputs
- Measures preservation of attention patterns
- Calculates overall function preservation scores

Example usage:
```python
from sentinel.plasticity.function_tracking import FunctionTracker, FunctionTrackingConfig

# Create configuration
config = FunctionTrackingConfig(
    output_dir="./output/function_tracking",
    experiment_name="function_preservation_study"
)

# Initialize tracker
tracker = FunctionTracker(config)

# Track function preservation between two model versions
test_prompts = ["The transformer architecture allows", "Neural networks consist of"]
tracker.track_function(
    model_before, 
    model_after, 
    test_prompts, 
    tokenizer, 
    cycle_idx=1, 
    cycle_name="After_Pruning"
)

# Create summary report
tracker.create_summary_report()

# Compare multiple cycles against a reference
tracker.compare_multiple_cycles("Initial")
```

### 3. Stress Protocols (`stress_protocols.py`)

Implements methods for testing neural plasticity under stress, including conflicting tasks, targeted pruning, and damage recovery. This module helps understand model resilience and adaptation capabilities.

Key features:
- Task Alternation Protocol: Tests adaptation to conflicting task demands
- Conflict Pruning Protocol: Tests recovery from targeted pruning of important heads
- Targeted Damage Protocol: Tests resilience to damage in specific layers
- Measures recovery rates and adaptation under stress

Example usage:
```python
from sentinel.plasticity.stress_protocols import (
    TaskAlternationProtocol, StressProtocolConfig, run_plasticity_stress_loop
)

# Create configuration
config = StressProtocolConfig(
    output_dir="./output/stress_protocols",
    experiment_name="task_alternation_test",
    protocol_type="task_alternation",
    cycles=5,
    tasks=["general", "scientific"]
)

# Create protocol
protocol = TaskAlternationProtocol(config)

# Run protocol
results = protocol.run_protocol(
    model,
    dataloaders,
    fine_tuning_fn=my_fine_tuning_function
)

# Or use convenience function
results = run_plasticity_stress_loop(
    model,
    ["general", "scientific"],
    dataloaders,
    config,
    fine_tuning_fn=my_fine_tuning_function
)
```

### 4. Entropy Rhythm Plot (`entropy_rhythm_plot.py`)

Creates EEG-like visualizations of attention entropy patterns over time, revealing rhythmic adaptation in transformer models. This module helps visualize the "brain waves" of model adaptation.

Key features:
- Static EEG-like rhythm plots showing attention entropy over time
- Animated visualizations showing entropy evolution
- Delta heatmaps comparing entropy changes between cycles
- Time-series plots of entropy patterns

Example usage:
```python
from sentinel.visualization.entropy_rhythm_plot import (
    plot_entropy_rhythm,
    create_animated_entropy_rhythm,
    create_entropy_delta_heatmap,
    plot_entropy_rhythm_from_file
)

# Create static rhythm plot
fig = plot_entropy_rhythm(
    entropy_df,
    save_path="rhythm_plot.png",
    normalize=True,
    smooth_window=2,
    title="Entropy Rhythm Across Cycles"
)

# Create animated visualization
create_animated_entropy_rhythm(
    entropy_df,
    save_path="rhythm_animation.mp4",
    fps=5,
    normalize=True
)

# Create delta heatmap
create_entropy_delta_heatmap(
    entropy_df,
    save_path="delta_heatmap.png",
    title="Entropy Changes Between Cycles"
)

# Or load directly from journal file
plot_entropy_rhythm_from_file(
    "path/to/entropy_journal.jsonl",
    save_path="rhythm_from_file.png"
)
```

## Complete Demo

A complete demonstration is available in `scripts/neural_plasticity_demo.py`. This script shows how all components work together to study neural plasticity in transformer models.

Run the demo with:
```bash
python scripts/neural_plasticity_demo.py \
    --model_name distilgpt2 \
    --output_dir ./output \
    --cycles 5 \
    --steps_per_cycle 50 \
    --run_stress_protocol
```

Options:
- `--model_name`: Hugging Face model name (default: distilgpt2)
- `--output_dir`: Output directory (default: ./output)
- `--cycles`: Number of plasticity cycles (default: 5)
- `--steps_per_cycle`: Training steps per cycle (default: 50)
- `--batch_size`: Batch size for training (default: 4)
- `--device`: Device (cuda or cpu)
- `--run_stress_protocol`: Run stress protocol after cycles
- `--stress_cycles`: Number of stress cycles (default: 3)

## Scientific Applications

The Neural Plasticity Tracking System enables several scientific applications:

1. **Studying adaptation to conflicting tasks**: How do models balance competing demands? Do they develop specialized subsystems?

2. **Measuring resilience to structural changes**: How well do models recover from pruning or other architectural modifications?

3. **Observing emergence of specialization**: How do attention heads develop specialized functions over time?

4. **Identifying rhythm patterns in learning**: Are there cyclical patterns in how models learn and adapt?

5. **Comparing plasticity across model scales**: How does plasticity change with model size and architecture?

## Contribution

This system represents a shift from static optimization to dynamic intelligence, treating models as subjects for scientific study rather than just engineering artifacts. It enables researchers to understand the temporal dimension of neural network intelligence, observing how capabilities emerge, adapt, and evolve over time.

By treating neural networks as systems with lifecycles, we can study fundamental questions about adaptation, resilience, and the dynamics of intelligence itself.