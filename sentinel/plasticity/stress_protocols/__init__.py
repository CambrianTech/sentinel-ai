#!/usr/bin/env python
"""
Stress Protocols for Neural Plasticity Testing

This package provides protocols for stress testing neural plasticity in transformer models,
including task alternation, memory stress tests, and conflicting objective tests.
"""

from sentinel.plasticity.stress_protocols.task_suite import (
    TaskSuite, TaskConfig, TaskExample,
    create_diverse_task_suite,
    create_memory_stress_task,
    create_conflicting_tasks
)

from sentinel.plasticity.stress_protocols.task_alternation import (
    TaskAlternationConfig, TaskAlternationProtocol,
    run_diverse_task_alternation, run_conflicting_task_alternation
)

__all__ = [
    'TaskSuite', 'TaskConfig', 'TaskExample',
    'create_diverse_task_suite', 'create_memory_stress_task', 'create_conflicting_tasks',
    'TaskAlternationConfig', 'TaskAlternationProtocol',
    'run_diverse_task_alternation', 'run_conflicting_task_alternation'
]