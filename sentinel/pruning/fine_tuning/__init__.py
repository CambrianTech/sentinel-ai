"""
Fine-tuning utilities for pruned transformer models.

This module provides specialized fine-tuning capabilities for models after pruning,
with a focus on maintaining or improving performance despite having fewer parameters.
"""

from sentinel.pruning.fine_tuning.fine_tuner import FineTuner
from sentinel.pruning.fine_tuning.fine_tuner_consolidated import ConsolidatedFineTuner
from sentinel.pruning.fine_tuning.fine_tuner_improved import ImprovedFineTuner