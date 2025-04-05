#!/usr/bin/env python
"""
Basic Test for Stress Protocols

This script tests the basic functionality of the stress protocols module without
requiring any external dependencies like transformers.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Step 1: Importing TaskSuite, TaskConfig, TaskExample...")
from sentinel.plasticity.stress_protocols.task_suite import (
    TaskSuite, TaskConfig, TaskExample
)
print("Success!")

print("\nStep 2: Creating a simple task suite...")
# Create a simple task config
simple_task = TaskConfig(
    name="simple_task",
    description="A simple test task",
    metric="accuracy",
    examples=[
        TaskExample(
            input_text="What is 1+1?",
            expected_output="2"
        ),
        TaskExample(
            input_text="What is the capital of France?",
            expected_output="Paris"
        )
    ]
)

# Create a task suite
task_suite = TaskSuite(
    name="test_suite",
    tasks=[simple_task],
    device="cpu"
)
print("Success!")

print("\nStep 3: Verify task suite functionality...")
# Verify task retrieval
task = task_suite.get_task("simple_task")
print(f"Task name: {task.name}")
print(f"Task description: {task.description}")
print(f"Task metric: {task.metric}")
print(f"Number of examples: {len(task.examples)}")
print(f"First example input: {task.examples[0].input_text}")
print(f"First example output: {task.examples[0].expected_output}")
print("Success!")

print("\nAll basic stress protocol tests passed!")

if __name__ == "__main__":
    pass