# Testing with Mocks Guide

This guide explains how to test components that depend on external libraries like `transformers` and `datasets` without triggering import conflicts.

## When to Use Mock Testing

Mock testing is particularly useful when:

1. You want to test components that rely on `transformers` or `datasets` libraries
2. You need to test task protocols without requiring a full model
3. You want fast, reliable tests that don't depend on external APIs
4. You're writing unit tests for components that shouldn't need actual model execution

## Approaches to Mock Testing

We provide several approaches for mock testing, from simplest to most complex:

### 1. Isolated Mock Testing (Recommended)

The simplest and most reliable approach is to create completely isolated tests that don't import any problematic modules:

```python
#!/usr/bin/env python
# Create mock modules before any imports
class MockModule:
    def __init__(self, name):
        self.__name__ = name
        
    def __getattr__(self, name):
        return MockModule(f"{self.__name__}.{name}")

# Mock transformers and datasets modules
sys.modules['transformers'] = MockModule('transformers')
sys.modules['datasets'] = MockModule('datasets')

# Define your test functions using local definitions of the classes you need
```

See `scripts/isolated_mock_test.py` for a complete example.

### 2. Simple Mock Testing

For tests that need a bit more structure, you can use the `simple_test_with_mock.py` approach:

```python
# Set up mock modules
class MockModule:
    def __init__(self, name):
        self.__name__ = name
        
    def __getattr__(self, name):
        return MockModule(f"{self.__name__}.{name}")

# Mock transformers and datasets modules
sys.modules['transformers'] = MockModule('transformers')
sys.modules['datasets'] = MockModule('datasets')

# Now define your own test versions of the classes
@dataclass
class TaskExample:
    input_text: str
    expected_output: Optional[str] = None
    # ...

# Create and test instances
example = TaskExample(input_text="test", expected_output="result")
```

### 3. Using the Mock Module

For more sophisticated testing, you can use our `sentinel.mock` module:

```python
# Import and set up mocks before any other imports
from sentinel.mock.transformer_mocks import setup_transformer_mocks
from sentinel.mock.dataset_mocks import setup_dataset_mocks

setup_transformer_mocks()
setup_dataset_mocks()

# Now you can safely import from transformers and datasets
from transformers import PreTrainedTokenizer, PreTrainedModel
from datasets import Dataset

# Your tests here
```

**NOTE**: This approach is still experimental and may not work in all cases due to import order issues.

## Best Practices

1. **Set up mocks first**: Always set up mock modules before importing anything that might import the real modules.

2. **Avoid imports**: In isolated tests, avoid importing from your main codebase to prevent triggering problematic imports.

3. **Replicate minimal functionality**: Only mock the functionality you actually need for your tests.

4. **Test edge cases**: Include tests for error conditions and edge cases to ensure your code handles them correctly.

## Example: Testing Task Suite

Here's a complete example of testing the task suite with mocks:

```python
#!/usr/bin/env python
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Mock modules
sys.modules['transformers'] = type('transformers', (), {})
sys.modules['datasets'] = type('datasets', (), {})

# Define classes
@dataclass
class TaskExample:
    input_text: str
    expected_output: Optional[str] = None

@dataclass
class TaskConfig:
    name: str
    description: str
    metric: str
    examples: List[TaskExample] = field(default_factory=list)

class TaskSuite:
    def __init__(self, name, tasks, device="cpu"):
        self.name = name
        self.tasks = {task.name: task for task in tasks}
        
    def get_task(self, task_name):
        return self.tasks[task_name]
    
    def get_task_names(self):
        return list(self.tasks.keys())

# Test code
example = TaskExample("What is 1+1?", "2")
config = TaskConfig("math", "Math questions", "accuracy", [example])
suite = TaskSuite("test_suite", [config])

assert suite.get_task("math").examples[0].input_text == "What is 1+1?"
print("All tests passed!")
```

## Troubleshooting

If you encounter import issues:

1. Check if your test is importing anything that might trigger imports of the real modules
2. Try using the isolated approach with minimal imports
3. Use `python -v` to see detailed import traces and identify problematic imports