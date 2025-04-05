"""
Sentinel-AI Utilities Module

This module provides various utility functions and classes used across the
Sentinel-AI system, including metrics tracking, checkpointing, model wrappers,
and training utilities.
"""

# Import key utilities for easy access
# Use try/except to handle modules that might not be fully moved yet
try:
    from sentinel.utils.checkpoint import save_checkpoint, load_checkpoint
except ImportError:
    pass

try:
    from sentinel.utils.metrics import calculate_metrics, log_metrics
except ImportError:
    pass

try:
    from sentinel.utils.model_wrapper import ModelWrapper
except ImportError:
    pass

try:
    from sentinel.utils.progress_tracker import ProgressTracker
except ImportError:
    pass

try:
    from sentinel.utils.head_lr_manager import HeadLRManager
except ImportError:
    pass

try:
    from sentinel.utils.generation_wrapper import GenerationWrapper
except ImportError:
    pass

try:
    from sentinel.utils.metrics_logger import MetricsLogger
except ImportError:
    pass

try:
    from sentinel.utils.training import Trainer
except ImportError:
    pass

try:
    from sentinel.utils.dynamic_architecture import DynamicArchitecture
except ImportError:
    pass

try:
    from sentinel.utils.adaptive import AdaptivePlasticitySystem, run_adaptive_system
except ImportError:
    pass

try:
    from sentinel.utils.head_metrics import (
        compute_attention_entropy,
        compute_head_importance,
        compute_gradient_norms,
        visualize_head_metrics,
        recommend_pruning_growth
    )
except ImportError:
    pass

# For backward compatibility
try:
    import sys
    import warnings
    from importlib import import_module
    from functools import wraps
    
    # Define a function to create a proxy class that imports from the new location
    def _create_proxy_class(old_module_name, class_name, new_module_name):
        warnings.warn(
            f"{old_module_name}.{class_name} is deprecated. "
            f"Please use {new_module_name}.{class_name} instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Return the actual class from the new location
        new_module = import_module(new_module_name)
        return getattr(new_module, class_name)
    
    # Define a function to create a proxy function that imports from the new location
    def _create_proxy_function(old_module_name, func_name, new_module_name):
        # Get the actual function from the new location
        new_module = import_module(new_module_name)
        real_func = getattr(new_module, func_name)
        
        # Create a wrapper function that warns about deprecation
        @wraps(real_func)
        def proxy_func(*args, **kwargs):
            warnings.warn(
                f"{old_module_name}.{func_name} is deprecated. "
                f"Please use {new_module_name}.{func_name} instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return real_func(*args, **kwargs)
        
        return proxy_func
    
    # Create and install proxy modules to maintain backward compatibility
    # This allows code that imports from old locations to continue working
    sys.modules['utils.checkpoint'] = type('ProxyModule', (), {
        'save_checkpoint': _create_proxy_function('utils.checkpoint', 'save_checkpoint', 'sentinel.utils.checkpoint'),
        'load_checkpoint': _create_proxy_function('utils.checkpoint', 'load_checkpoint', 'sentinel.utils.checkpoint'),
        '__name__': 'utils.checkpoint',
    })
    
    sys.modules['utils.metrics'] = type('ProxyModule', (), {
        'calculate_metrics': _create_proxy_function('utils.metrics', 'calculate_metrics', 'sentinel.utils.metrics'),
        'log_metrics': _create_proxy_function('utils.metrics', 'log_metrics', 'sentinel.utils.metrics'),
        '__name__': 'utils.metrics',
    })
    
    sys.modules['utils.model_wrapper'] = type('ProxyModule', (), {
        'ModelWrapper': _create_proxy_class('utils.model_wrapper', 'ModelWrapper', 'sentinel.utils.model_wrapper'),
        '__name__': 'utils.model_wrapper',
    })
    
    sys.modules['utils.progress_tracker'] = type('ProxyModule', (), {
        'ProgressTracker': _create_proxy_class('utils.progress_tracker', 'ProgressTracker', 'sentinel.utils.progress_tracker'),
        '__name__': 'utils.progress_tracker',
    })
    
    sys.modules['utils.head_lr_manager'] = type('ProxyModule', (), {
        'HeadLRManager': _create_proxy_class('utils.head_lr_manager', 'HeadLRManager', 'sentinel.utils.head_lr_manager'),
        '__name__': 'utils.head_lr_manager',
    })
    
    sys.modules['utils.generation_wrapper'] = type('ProxyModule', (), {
        'GenerationWrapper': _create_proxy_class('utils.generation_wrapper', 'GenerationWrapper', 'sentinel.utils.generation_wrapper'),
        '__name__': 'utils.generation_wrapper',
    })
    
    sys.modules['utils.metrics_logger'] = type('ProxyModule', (), {
        'MetricsLogger': _create_proxy_class('utils.metrics_logger', 'MetricsLogger', 'sentinel.utils.metrics_logger'),
        '__name__': 'utils.metrics_logger',
    })
    
    sys.modules['utils.training'] = type('ProxyModule', (), {
        'Trainer': _create_proxy_class('utils.training', 'Trainer', 'sentinel.utils.training'),
        '__name__': 'utils.training',
    })
    
    sys.modules['utils.dynamic_architecture'] = type('ProxyModule', (), {
        'DynamicArchitecture': _create_proxy_class('utils.dynamic_architecture', 'DynamicArchitecture', 'sentinel.utils.dynamic_architecture'),
        '__name__': 'utils.dynamic_architecture',
    })
    
except ImportError:
    # Module not available, will use the new location
    pass