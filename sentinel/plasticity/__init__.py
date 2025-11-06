"""
Neural Plasticity Module for Sentinel-AI.

This module contains tools for studying neural plasticity through pruning, 
fine-tuning, and adaptation tracking in transformer models. It also includes
a neural defragmentation system with sleep-cycle inspired maintenance phases.
It provides stress testing protocols to measure plasticity under various conditions.
"""

from .plasticity_loop import (
    PlasticityTracker,
    AdaptiveFinetuner,
    PlasticityExperiment,
    run_plasticity_experiment
)

from .defrag_heads import (
    HeadDefragmenter,
    DefragConfiguration,
    defrag_model
)

from .sleep_cycle import (
    TransformerSleepCycle,
    SleepCycleConfig,
    CyclePhase,
    create_sleep_cycle
)

from .function_tracking import (
    ModelProbe, 
    FunctionTracker, 
    track_function
)

from .entropy_journal import (
    EntropyJournal,
    EntropyJournalConfig,
    record_entropy
)

# Import stress protocols
from .stress_protocols import (
    TaskSuite, TaskConfig, TaskExample,
    create_diverse_task_suite,
    create_memory_stress_task,
    create_conflicting_tasks,
    TaskAlternationConfig, TaskAlternationProtocol,
    run_diverse_task_alternation, run_conflicting_task_alternation
)

__all__ = [
    # Plasticity Loop
    'PlasticityTracker',
    'AdaptiveFinetuner',
    'PlasticityExperiment',
    'run_plasticity_experiment',
    
    # Neural Defragmentation
    'HeadDefragmenter',
    'DefragConfiguration',
    'defrag_model',
    
    # Sleep Cycle
    'TransformerSleepCycle',
    'SleepCycleConfig',
    'CyclePhase',
    'create_sleep_cycle',
    
    # Function Tracking
    'ModelProbe',
    'FunctionTracker',
    'track_function',
    
    # Entropy Journal
    'EntropyJournal',
    'EntropyJournalConfig',
    'record_entropy',
    
    # Stress Protocols
    'TaskSuite', 'TaskConfig', 'TaskExample',
    'create_diverse_task_suite', 'create_memory_stress_task', 'create_conflicting_tasks',
    'TaskAlternationConfig', 'TaskAlternationProtocol',
    'run_diverse_task_alternation', 'run_conflicting_task_alternation'
]