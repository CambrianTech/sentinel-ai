# Neural Plasticity Stress Test Protocols

This document describes the neural plasticity stress test protocols implemented in Sentinel-AI. These protocols are designed to study how transformer models adapt to changing task demands over time.

## Overview

The stress test protocols system consists of:

1. **Task Suites**: Collections of tasks with standardized interfaces for diverse capabilities
2. **Task Alternation Protocol**: Protocol for switching between tasks to test adaptation
3. **Visualization and Analysis Tools**: Tools for measuring forgetting and recovery rates

## Components

### Task Suite (`task_suite.py`)

The Task Suite module provides a standardized way to define collections of tasks for stress testing neural plasticity. It includes:

- `TaskExample`: A single task example with input, expected output, and metadata
- `TaskConfig`: Configuration for a specific task with metrics and examples
- `TaskSuite`: Collection of tasks with methods for creating dataloaders and evaluation

Predefined task suites:
- **Diverse Tasks**: Commonsense QA, summarization, code completion, NLI
- **Memory Stress**: Long context recall, key-value memory tasks
- **Conflicting Tasks**: Pairs of tasks with contradictory objectives

### Task Alternation Protocol (`task_alternation.py`)

The Task Alternation Protocol implements a rigorous approach for testing neural plasticity by:

1. Training on task A for N epochs
2. Evaluating on task A to establish baseline
3. Switching to task B for N epochs
4. Evaluating on both task A and B to measure:
   - Learning efficiency on task B
   - Forgetting rate on task A
   - Function preservation across tasks

This protocol provides quantifiable metrics for:
- Forgetting rates: How much performance drops when switching tasks
- Recovery rates: How quickly performance recovers when returning to a task
- Adaptation efficiency: How well the model adapts to new tasks

### Runner Script (`run_stress_test_protocol.py`)

The runner script provides a command-line interface for running stress test protocols across multiple models, making it easy to compare adaptation capabilities.

## Usage Examples

### Basic Test

```python
from sentinel.plasticity.stress_protocols import (
    create_diverse_task_suite,
    TaskAlternationConfig,
    TaskAlternationProtocol
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Create task suite
task_suite = create_diverse_task_suite()

# Configure protocol
config = TaskAlternationConfig(
    tasks=["commonsense_qa", "summarization"],
    cycles=3,
    epochs_per_task=1,
    output_dir="results/task_alternation"
)

# Run protocol
protocol = TaskAlternationProtocol(task_suite, config)
results = protocol.run_protocol(model, tokenizer)
```

### Command-Line Usage

```bash
# Run stress tests on multiple models
python scripts/run_stress_test_protocol.py \
    --models distilgpt2,gpt2,facebook/opt-125m \
    --protocol diverse \
    --output_dir outputs/stress_tests \
    --cycles 3
```

### Quick Test Run

```bash
# Run a minimal test on a small model
python scripts/test_task_alternation.py \
    --model distilgpt2 \
    --test_type diverse \
    --output_dir outputs/test_run
```

## Key Metrics

The protocols track several key metrics:

1. **Forgetting Rate**: Percentage of performance lost on a task after training on different tasks
2. **Recovery Rate**: How quickly performance returns when coming back to a previously learned task
3. **Adaptation Efficiency**: How many epochs are needed to reach good performance on a new task
4. **Cross-Task Transfer**: Whether learning one task helps or hinders performance on other tasks

## Visualization

The system generates several visualizations:

1. **Task Performance Plot**: Shows performance on each task over time
2. **Forgetting Rate Plot**: Shows how much performance is lost when switching tasks
3. **Recovery Rate Plot**: Shows how quickly performance recovers when returning to tasks
4. **Comparative Visualizations**: When running multi-model tests, shows comparison charts

## Integration with Other Modules

The stress protocols can integrate with other neural plasticity modules:

- **Entropy Journal**: Track attention entropy patterns during task alternation
- **Function Tracking**: Measure functional similarity between model versions
- **RL Controller**: Optimize pruning strategies based on adaptation performance

## Next Steps and Recommendations

1. **Import Resolution**: Resolve the import conflict between `sentinel_data` and the Hugging Face datasets library
2. **Task Suite Expansion**: Add more diverse and challenging tasks to the task suites
3. **Integration Testing**: Test integration with the entropy journal and function tracker
4. **Hyperparameter Tuning**: Experiment with different cycle lengths and learning rates
5. **Comparative Analysis**: Run tests across model families to compare adaptation capabilities

## Technical Notes

- The protocol automatically handles checkpointing and results saving
- All metrics are logged in JSON format for easy analysis
- The system includes delta measurement to track changes between cycles
- Visualization tools are designed to highlight important patterns in adaptation