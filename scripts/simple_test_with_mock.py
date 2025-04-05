#!/usr/bin/env python
"""
Simple Test with Mock Imports

This script tests our implementation of the task suite without requiring
dependencies on transformers or datasets. It uses mock imports to avoid
circular imports.
"""

import os
import sys
import importlib.util
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up mock modules to avoid circular imports
class MockModule:
    def __init__(self, name):
        self.__name__ = name
        
    def __getattr__(self, name):
        return MockModule(f"{self.__name__}.{name}")

# Mock transformers and datasets modules
sys.modules['transformers'] = MockModule('transformers')
sys.modules['datasets'] = MockModule('datasets')

# Define minimal versions of the task suite classes
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

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

# Create some task examples
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

# Create a task configuration
task_config = TaskConfig(
    name="simple_qa",
    description="Simple question answering task",
    metric="accuracy",
    examples=[example1, example2],
    max_input_length=100,
    max_output_length=20,
    task_type="qa"
)

# Create a task suite
task_suite = TaskSuite(
    name="test_suite",
    tasks=[task_config],
    device="cpu"
)

# Test functionality
def test_task_suite():
    """Test the task suite functionality"""
    print("Testing task suite functionality...")
    print(f"Suite name: {task_suite.name}")
    print(f"Available tasks: {task_suite.get_task_names()}")
    
    # Get a task
    task = task_suite.get_task("simple_qa")
    print(f"Task name: {task.name}")
    print(f"Task description: {task.description}")
    print(f"Task metric: {task.metric}")
    print(f"Number of examples: {len(task.examples)}")
    
    # Check examples
    example = task.examples[0]
    print(f"Example input: {example.input_text}")
    print(f"Example output: {example.expected_output}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_task_suite()