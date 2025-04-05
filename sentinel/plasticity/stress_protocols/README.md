# Neural Plasticity Stress Protocols

This package provides protocols for stress testing neural plasticity in transformer models. The protocols are designed to test how well models adapt to changing task demands, maintain prior capabilities, and recover after stress periods.

## Components

The stress protocol system consists of:

1. **Task Suites**: Collections of tasks with standardized interfaces for diverse capabilities
2. **Task Alternation Protocol**: Protocol for switching between tasks to test adaptation
3. **Visualization Tools**: Tools for visualizing forgetting and recovery patterns

## Task Suite Types

The package provides several pre-defined task suites:

### 1. Diverse Task Suite

Tasks with different skills and objectives:
- Commonsense QA
- Summarization
- Code completion
- Natural language inference

### 2. Memory Stress Suite

Tasks designed to stress model memory capabilities:
- Long context completion (find information from the beginning)
- Key-value recall across long sequences

### 3. Conflicting Task Suite

Tasks with competing or contradictory objectives:
- Standard text completion
- Reversed text completion
- Literal interpretation tasks
- Idiomatic interpretation tasks

## Task Alternation Protocol

The Task Alternation Protocol tests neural plasticity by:

1. Training on task A for N epochs
2. Evaluating on task A to establish baseline
3. Switching to task B for N epochs
4. Evaluating on both task A and B to measure:
   - How well the model learns task B
   - How much of task A capability is preserved
5. Repeating with additional tasks or cycles

The protocol quantifies:
- Learning efficiency per task
- Catastrophic forgetting effects
- Recovery rate when returning to previous tasks
- Function preservation across task boundaries

## Key Metrics

- **Forgetting Rate**: How much performance on a task drops after training on different tasks
- **Recovery Rate**: How quickly performance returns when training on a task again
- **Cross-Task Transfer**: How training on one task affects performance on others

## Usage Examples

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentinel.plasticity.stress_protocols import (
    create_diverse_task_suite,
    TaskAlternationConfig,
    TaskAlternationProtocol
)

# Load model and tokenizer
model_name = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create task suite
task_suite = create_diverse_task_suite()

# Configure protocol
config = TaskAlternationConfig(
    tasks=["commonsense_qa", "summarization", "code_completion"],
    cycles=3,
    epochs_per_task=1,
    output_dir="results/task_alternation"
)

# Create and run protocol
protocol = TaskAlternationProtocol(task_suite, config)
results = protocol.run_protocol(model, tokenizer)
```

### Running with Predefined Functions

```python
from sentinel.plasticity.stress_protocols import run_diverse_task_alternation

# Run with predefined setup
results = run_diverse_task_alternation(
    model=model,
    tokenizer=tokenizer,
    output_dir="results/diverse_alternation",
    cycles=3,
    epochs_per_task=1
)
```

### Multi-Model Comparison

```bash
# Run from command line
python scripts/run_stress_test_protocol.py \
    --models distilgpt2,gpt2,facebook/opt-125m \
    --protocol diverse \
    --output_dir outputs/model_comparison \
    --cycles 3
```

## Creating Custom Task Suites

You can create custom task suites for your specific needs:

```python
from sentinel.plasticity.stress_protocols import TaskSuite, TaskConfig, TaskExample

# Create custom tasks
custom_task = TaskConfig(
    name="my_custom_task",
    description="Custom domain-specific task",
    metric="accuracy",
    task_type="generation",
    examples=[
        TaskExample(
            input_text="Custom prompt 1",
            expected_output="Expected response 1"
        ),
        # Add more examples...
    ]
)

# Create custom suite
custom_suite = TaskSuite(
    name="custom_domain_tasks",
    tasks=[custom_task]
)
```

## Visualizing Results

The protocol automatically generates visualizations:

1. Task performance over time
2. Forgetting rates by task
3. Recovery rates across cycles
4. Multi-model comparisons (when using the comparison script)

Results and visualizations are saved to the specified output directory.

## Integration with Other Modules

The stress protocols integrate with other neural plasticity modules:

```python
from sentinel.plasticity import (
    EntropyJournal, FunctionTracker, TaskAlternationProtocol
)

# Track entropy changes during task alternation
journal = EntropyJournal()
function_tracker = FunctionTracker()

# Add to your custom training function
def custom_training(model, dataloader, task_name, epochs):
    # ... training code ...
    
    # Track entropy after each epoch
    journal.record_model_state(model, dataloader, epoch)
    
    # Track function preservation
    function_tracker.track_function(model_before, model, prompts)
    
    return metrics

# Use in protocol
protocol = TaskAlternationProtocol(task_suite, config)
results = protocol.run_protocol(model, tokenizer, fine_tuning_fn=custom_training)
```

## Future Development

Planned enhancements:
- More diverse and challenging task suites
- Integration with reinforcement learning controllers
- Adaptive difficulty based on model performance
- Longitudinal tracking across experiments