#!/usr/bin/env python
"""
Minimal Test for Task Protocols

This script defines and tests minimal versions of the task protocol classes
to verify they work without any external dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Define minimal versions of the classes for testing
@dataclass
class TaskExample:
    """A single task example with input, expected output, and metadata"""
    input_text: str
    expected_output: Optional[str] = None
    task_type: str = "generation"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskConfig:
    """Configuration for a specific task"""
    name: str
    description: str
    metric: str  # e.g., "accuracy", "f1", "rouge", "bleu", "perplexity"
    examples: List[TaskExample] = field(default_factory=list)
    max_input_length: int = 512
    max_output_length: int = 128
    task_type: str = "generation"  # generation, classification, qa, etc.

class TaskSuite:
    """Collection of tasks for stress testing"""
    def __init__(
        self,
        name: str,
        tasks: List[TaskConfig],
        device: str = "cpu"
    ):
        self.name = name
        self.tasks = {task.name: task for task in tasks}
        self.device = device
    
    def get_task(self, task_name: str) -> TaskConfig:
        """Get a task by name"""
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found in suite. Available tasks: {list(self.tasks.keys())}")
        return self.tasks[task_name]
    
    def get_task_names(self) -> List[str]:
        """Get list of available task names"""
        return list(self.tasks.keys())

# Run tests
def run_tests():
    print("Step 1: Creating task examples...")
    example1 = TaskExample(
        input_text="What is 1+1?",
        expected_output="2",
        task_type="qa"
    )
    example2 = TaskExample(
        input_text="What is the capital of France?",
        expected_output="Paris",
        task_type="qa"
    )
    
    print(f"Example 1 input: {example1.input_text}")
    print(f"Example 1 output: {example1.expected_output}")
    print(f"Example 2 input: {example2.input_text}")
    print(f"Example 2 output: {example2.expected_output}")
    print("Success!")
    
    print("\nStep 2: Creating task configuration...")
    task_config = TaskConfig(
        name="simple_qa",
        description="Simple question answering task",
        metric="accuracy",
        examples=[example1, example2],
        max_input_length=100,
        max_output_length=20,
        task_type="qa"
    )
    
    print(f"Task name: {task_config.name}")
    print(f"Task description: {task_config.description}")
    print(f"Task metric: {task_config.metric}")
    print(f"Number of examples: {len(task_config.examples)}")
    print("Success!")
    
    print("\nStep 3: Creating task suite...")
    task_suite = TaskSuite(
        name="test_suite",
        tasks=[task_config],
        device="cpu"
    )
    
    print(f"Suite name: {task_suite.name}")
    print(f"Available tasks: {task_suite.get_task_names()}")
    
    retrieved_task = task_suite.get_task("simple_qa")
    print(f"Retrieved task name: {retrieved_task.name}")
    print(f"Retrieved task examples: {len(retrieved_task.examples)}")
    print("Success!")
    
    print("\nAll minimal task suite tests passed!")

if __name__ == "__main__":
    run_tests()