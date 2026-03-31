"""
ForgeAlloy stage executors — one class per alloy stage type.

Each executor implements StageExecutor.execute(ctx) and transforms the ForgeContext.
The registry maps stage type strings to executor classes.

Usage:
    from stages import STAGE_EXECUTORS, create_executor
    executor = create_executor(stage_config)
    ctx = executor.execute(ctx)
"""

from .base import StageExecutor, ForgeContext, create_executor
from .registry import STAGE_EXECUTORS

__all__ = ['StageExecutor', 'ForgeContext', 'create_executor', 'STAGE_EXECUTORS']
