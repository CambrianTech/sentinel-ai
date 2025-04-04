"""
DEPRECATED: This module has moved to sentinel.pruning.strategies.base
This import stub will be removed in a future version.
"""
import warnings
warnings.warn(
    "Importing from utils.pruning.strategies is deprecated. "
    "Use sentinel.pruning.strategies.base instead.",
    DeprecationWarning, 
    stacklevel=2
)

from sentinel.pruning.strategies.base import *
