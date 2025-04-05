"""
Neural Plasticity Module for Sentinel-AI.

This module contains tools for studying neural plasticity through pruning, 
fine-tuning, and adaptation tracking in transformer models. It also includes
a neural defragmentation system with sleep-cycle inspired maintenance phases.
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
    'create_sleep_cycle'
]