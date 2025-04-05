"""
Sentinel-AI: A framework for neural plasticity and adaptive transformer architectures

Sentinel-AI is a research framework for studying how transformer models
can adapt their structure while preserving function, through cycles of
pruning, recovery, and reorganization.
"""

__version__ = "0.2.0"

SENTINEL_MANIFESTO = """
Sentinel-AI preserves functional continuity through structural adaptation, 
pruning, and regrowth. It is a dynamic system that studies how neural networks 
can maintain capabilities while evolving their architecture.
"""

# Include metadata in checkpoint and logs
SENTINEL_METADATA = {
    "name": "Sentinel-AI",
    "version": __version__,
    "manifesto": SENTINEL_MANIFESTO,
    "repository": "https://github.com/CambrianTech/sentinel-ai"
}

# Import main modules
from sentinel.models import adaptive_transformer
from sentinel.controller import controller_manager
from sentinel.pruning import pruning_module
from sentinel.plasticity import plasticity_loop, sleep_cycle, defrag_heads

# Import new plasticity tracking modules
from sentinel.plasticity import entropy_journal, function_tracking, stress_protocols
from sentinel.visualization import entropy_rhythm_plot
from sentinel.plasticity.controller import rl_controller