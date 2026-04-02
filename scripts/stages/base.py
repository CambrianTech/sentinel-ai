"""Base classes for ForgeAlloy stage execution."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ForgeContext:
    """Shared state that flows through the pipeline.
    Each stage reads from and writes to this context."""
    model: object = None
    tokenizer: object = None
    model_name: str = ""
    output_dir: Path = field(default_factory=lambda: Path("output/forged"))
    alloy: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    baseline_ppl: float = 0.0
    final_ppl: float = 0.0
    cycle_results: list = field(default_factory=list)
    samples: dict = field(default_factory=dict)
    device: str = ""
    tier: str = ""
    load_4bit: bool = False

    # Internal state accumulated across stages
    hooks: list = field(default_factory=list)
    eval_results: list = field(default_factory=list)
    source_config: dict = field(default_factory=dict)
    layer_importance: list = field(default_factory=list)  # Per-layer importance from pruning → feeds quant


class StageExecutor:
    """Base class for stage executors. Each alloy stage type extends this."""

    def __init__(self, config: dict):
        self.config = config

    @property
    def stage_type(self) -> str:
        return self.config.get("type", "unknown")

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        """Execute this stage, transforming the context. Override in subclasses."""
        self.log(f"Stage type '{self.stage_type}' has no executor — skipping")
        return ctx

    def log(self, msg: str):
        print(f"  [{self.stage_type}] {msg}")


def create_executor(stage: dict) -> StageExecutor:
    """Create the appropriate executor for an alloy stage config."""
    from .registry import STAGE_EXECUTORS
    stype = stage.get("type", "unknown")
    cls = STAGE_EXECUTORS.get(stype, StageExecutor)
    return cls(stage)
