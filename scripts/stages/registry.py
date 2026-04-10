"""Stage executor registry — maps alloy stage types to executor classes.

Add a new alloy stage type → create executor class → register here → done.
The alloy_executor discovers stages from this registry.
"""

from .base import StageExecutor
from .input_stages import SourceConfigExecutor, ContextExtendExecutor, ModalityExecutor
from .transform_stages import (
    PruneExecutor, TrainExecutor, ExpertPruneExecutor,
    ExpertActivationProfileExecutor,
)
from .output_stages import QuantExecutor, PackageExecutor, EvalExecutor, DeliverExecutor, PublishExecutor, DeployExecutor

STAGE_EXECUTORS: dict[str, type[StageExecutor]] = {
    # Input stages (front of pipeline)
    "source-config": SourceConfigExecutor,
    "context-extend": ContextExtendExecutor,
    "modality": ModalityExecutor,

    # Transform stages (middle, cycled)
    "prune": PruneExecutor,
    "train": TrainExecutor,
    "lora": TrainExecutor,  # LoRA is a training variant — same executor for now
    "expert-activation-profile": ExpertActivationProfileExecutor,
    "expert-prune": ExpertPruneExecutor,

    # Output stages (end of pipeline)
    "quant": QuantExecutor,
    "package": PackageExecutor,
    "eval": EvalExecutor,
    "deliver": DeliverExecutor,
    "publish": PublishExecutor,  # deprecated — delegates to DeliverExecutor + still uploads
    "deploy": DeployExecutor,
}
