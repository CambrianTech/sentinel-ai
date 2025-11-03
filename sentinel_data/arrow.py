# sentinel_data/arrow.py
"""
Patched arrow module for HuggingFace datasets library.

This module provides a patched version of the datasets arrow module
to avoid the 'datasets' circular import.
"""

import logging
import itertools
from dataclasses import dataclass
from typing import Optional

import pyarrow as pa

import sentinel_data
from sentinel_data.table import table_cast

# Create logger without using datasets.utils.logging
logger = logging.getLogger("datasets")

class BuilderConfig:
    """Stub BuilderConfig class to avoid importing from datasets"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

@dataclass
class ArrowConfig(BuilderConfig):
    """Configuration for arrow reading/writing operations."""

    features: Optional[sentinel_data.features.Features] = None


def generate_batch_chunks(*iterables, batch_size: int):
    """
    Generate batches of objects from iterables.

    Example:

    >>> list(generate_batch_chunks([1, 2, 3, 4], [5, 6, 7, 8], batch_size=2))
    [([1, 2], [5, 6]), ([3, 4], [7, 8])]
    """
    # Use transpose to aggregate chunks by columns
    chunk_generator = itertools.zip_longest(*[iter(iterable) for iterable in iterables], fillvalue=None)
    chunk_aggregator = itertools.zip_longest(*[iter(chunk_generator)] * batch_size, fillvalue=None)

    # Tranpose chunks aggregated by rows
    transposed_chunk_aggregator = map(filter_none_values, map(transpose, chunk_aggregator))
    yield from filter_none_values(transposed_chunk_aggregator)


def transpose(list_of_tuples):
    """
    Transpose a list of tuples.

    Example:

    >>> list(transpose([(1, 2), (3, 4)]))
    [(1, 3), (2, 4)]
    """
    return zip(*list_of_tuples)


def filter_none_values(iterable):
    """Remove None values from the iterable."""
    return filter(lambda x: x is not None, iterable)