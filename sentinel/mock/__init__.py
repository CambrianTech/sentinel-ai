# sentinel/mock/__init__.py
"""
Mock modules for testing without external dependencies.

This module provides mock implementations of external dependencies like 
transformers and datasets to allow testing without circular imports.
"""

from sentinel.mock.transformer_mocks import (
    MockPreTrainedTokenizer,
    MockPreTrainedModel,
    MockTrainer,
    MockTrainingArguments,
    setup_transformer_mocks
)

from sentinel.mock.dataset_mocks import (
    MockDataset,
    MockFeatures,
    MockArrowTable,
    setup_dataset_mocks
)

__all__ = [
    'MockPreTrainedTokenizer',
    'MockPreTrainedModel',
    'MockTrainer',
    'MockTrainingArguments',
    'MockDataset',
    'MockFeatures',
    'MockArrowTable',
    'setup_transformer_mocks',
    'setup_dataset_mocks',
]