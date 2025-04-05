#!/usr/bin/env python
"""
Isolated Mock Test

This script demonstrates the mock module approach in complete isolation,
without importing any modules that might trigger problematic imports.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Create mock modules first, before any imports
class MockModule:
    def __init__(self, name):
        self.__name__ = name
        
    def __getattr__(self, name):
        return MockModule(f"{self.__name__}.{name}")

# Mock the transformers module
transformers_mock = MockModule("transformers")
transformers_mock.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {
    "__init__": lambda self, *args, **kwargs: None,
    "encode": lambda self, text, **kwargs: [1, 2, 3],
    "decode": lambda self, tokens, **kwargs: "Decoded text",
    "save_pretrained": lambda self, path: None
})
transformers_mock.PreTrainedModel = type("PreTrainedModel", (), {
    "__init__": lambda self, *args, **kwargs: None,
    "__call__": lambda self, **kwargs: type("Output", (), {"loss": 0.5, "logits": [[0.1, 0.2, 0.7]]}),
    "to": lambda self, device: self,
    "eval": lambda self: self,
    "train": lambda self: self,
    "generate": lambda self, **kwargs: [[1, 2, 3]],
    "save_pretrained": lambda self, path: None
})
transformers_mock.Trainer = type("Trainer", (), {
    "__init__": lambda self, *args, **kwargs: None,
    "train": lambda self, *args, **kwargs: {"training_loss": 0.5},
    "evaluate": lambda self, *args, **kwargs: {"eval_loss": 0.4}
})
transformers_mock.TrainingArguments = type("TrainingArguments", (), {
    "__init__": lambda self, *args, **kwargs: None
})

# Mock the datasets module
datasets_mock = MockModule("datasets")
datasets_mock.Dataset = type("Dataset", (), {
    "__init__": lambda self, *args, **kwargs: None,
    "__len__": lambda self: 10,
    "__getitem__": lambda self, idx: {"text": "Sample text"},
    "map": lambda self, *args, **kwargs: self,
    "filter": lambda self, *args, **kwargs: self
})
datasets_mock.Features = type("Features", (), {
    "__init__": lambda self, *args, **kwargs: None
})
datasets_mock.load_dataset = lambda *args, **kwargs: datasets_mock.Dataset()

# Register mock modules
sys.modules["transformers"] = transformers_mock
sys.modules["datasets"] = datasets_mock

# Define our task suite classes (copied to avoid importing)
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

    def create_dataloader(self, task_name, tokenizer, batch_size=4, shuffle=True):
        """Mock create_dataloader method"""
        # Return a mock dataloader
        return [({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}, 0) for _ in range(5)]
    
    def evaluate(self, task_name, model, tokenizer, device=None):
        """Mock evaluate method"""
        # Return mock evaluation metrics
        return {
            "score": 0.85,
            "metric": "accuracy"
        }

def test_mocks():
    """Test the mock implementations"""
    print("Testing mock implementations...")
    
    # Import from mock modules
    from transformers import PreTrainedTokenizer, PreTrainedModel, Trainer, TrainingArguments
    from datasets import Dataset, Features, load_dataset
    
    # Test transformers mocks
    tokenizer = PreTrainedTokenizer()
    model = PreTrainedModel()
    
    encoded = tokenizer.encode("This is a test")
    print(f"Encoded text: {encoded}")
    
    output = model(input_ids=[encoded])
    print(f"Model output loss: {output.loss}")
    
    # Test datasets mocks
    dataset = load_dataset("dummy")
    print(f"Dataset sample: {dataset[0]}")
    
    # Test task suite with mocks
    examples = [
        TaskExample(
            input_text="What is 1+1?",
            expected_output="2",
            task_type="qa"
        ),
        TaskExample(
            input_text="What is the capital of France?",
            expected_output="Paris",
            task_type="qa"
        )
    ]
    
    task_config = TaskConfig(
        name="simple_qa",
        description="Simple question answering task",
        metric="accuracy",
        examples=examples
    )
    
    task_suite = TaskSuite(
        name="test_suite",
        tasks=[task_config],
        device="cpu"
    )
    
    print(f"Task suite name: {task_suite.name}")
    print(f"Available tasks: {task_suite.get_task_names()}")
    
    task = task_suite.get_task("simple_qa")
    print(f"Task name: {task.name}")
    print(f"Task examples: {len(task.examples)}")
    
    # Test dataloader and evaluation
    dataloader = task_suite.create_dataloader("simple_qa", tokenizer, batch_size=4)
    print(f"Dataloader sample: {dataloader[0]}")
    
    eval_results = task_suite.evaluate("simple_qa", model, tokenizer)
    print(f"Evaluation results: {eval_results}")
    
    print("All mock tests passed!")
    
if __name__ == "__main__":
    test_mocks()