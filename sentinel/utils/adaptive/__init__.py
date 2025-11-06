"""
Adaptive Neural Plasticity System

This module implements a self-improving neural network architecture
that uses degeneration detection as a feedback mechanism to guide
pruning, growth, and learning cycles.
"""

from sentinel.utils.adaptive.adaptive_plasticity import AdaptivePlasticitySystem, run_adaptive_system

__all__ = ["AdaptivePlasticitySystem", "run_adaptive_system"]

# For backward compatibility
try:
    import sys
    import warnings
    from importlib import import_module
    
    # Import the actual implementations from the new location
    from sentinel.utils.adaptive.adaptive_plasticity import (
        AdaptivePlasticitySystem as _ActualAdaptivePlasticitySystem,
        run_adaptive_system as _actual_run_adaptive_system
    )
    
    # Create proxy classes and functions with deprecation warnings
    class DeprecatedAdaptivePlasticitySystem(_ActualAdaptivePlasticitySystem):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "utils.adaptive.AdaptivePlasticitySystem is deprecated. "
                "Please use sentinel.utils.adaptive.AdaptivePlasticitySystem instead.",
                DeprecationWarning,
                stacklevel=2
            )
            super().__init__(*args, **kwargs)
    
    def deprecated_run_adaptive_system(*args, **kwargs):
        warnings.warn(
            "utils.adaptive.run_adaptive_system is deprecated. "
            "Please use sentinel.utils.adaptive.run_adaptive_system instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return _actual_run_adaptive_system(*args, **kwargs)
    
    # Install proxy module for backward compatibility
    sys.modules['utils.adaptive'] = type('ProxyModule', (), {
        'AdaptivePlasticitySystem': DeprecatedAdaptivePlasticitySystem,
        'run_adaptive_system': deprecated_run_adaptive_system,
        '__name__': 'utils.adaptive',
    })
    
    # Also install the specific module
    sys.modules['utils.adaptive.adaptive_plasticity'] = type('ProxyModule', (), {
        'AdaptivePlasticitySystem': DeprecatedAdaptivePlasticitySystem,
        'run_adaptive_system': deprecated_run_adaptive_system,
        '__name__': 'utils.adaptive.adaptive_plasticity',
    })
    
except ImportError:
    # Module not available, will use the new location
    pass