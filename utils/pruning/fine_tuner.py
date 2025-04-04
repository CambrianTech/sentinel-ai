"""
DEPRECATED: This module has moved to sentinel.pruning.fine_tuning.base
This import stub will be removed in a future version.
"""
import warnings
warnings.warn(
    "Importing from utils.pruning.fine_tuner is deprecated. "
    "Use sentinel.pruning.fine_tuning.base instead.",
    DeprecationWarning, 
    stacklevel=2
)

from sentinel.pruning.fine_tuning.base import *
