"""
Sentinel-AI Pruning Module

This module provides utilities for pruning attention heads in transformer models,
identifying head importance, and managing growth and pruning strategies.
"""

# Import key components (as they are moved)
# from sentinel.utils.pruning.pruning_module import PruningModule
# from sentinel.utils.pruning.fixed_pruning_module import FixedPruningModule
# from sentinel.utils.pruning.strategies import get_strategy
# from sentinel.utils.pruning.growth import grow_attention_heads_gradually, determine_active_heads

# For backward compatibility
try:
    import sys
    import warnings
    from importlib import import_module
    
    # Define a function to create a proxy class that imports from the new location
    def _create_proxy_class(old_module_name, class_name, new_module_name):
        warnings.warn(
            f"{old_module_name}.{class_name} is deprecated. "
            f"Please use {new_module_name}.{class_name} instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Try to return the actual class from the new location
        try:
            new_module = import_module(new_module_name)
            return getattr(new_module, class_name)
        except (ImportError, AttributeError):
            # Module not available yet, raise a more helpful error
            raise ImportError(f"Could not import {class_name} from {new_module_name}")
    
    # Define a function to create a proxy function that imports from the new location
    def _create_proxy_function(old_module_name, func_name, new_module_name):
        # Try to get the actual function from the new location
        try:
            new_module = import_module(new_module_name)
            real_func = getattr(new_module, func_name)
            
            # Create a wrapper function that warns about deprecation
            def proxy_func(*args, **kwargs):
                warnings.warn(
                    f"{old_module_name}.{func_name} is deprecated. "
                    f"Please use {new_module_name}.{func_name} instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                return real_func(*args, **kwargs)
                
            return proxy_func
        except (ImportError, AttributeError):
            # Module not available yet, raise a more helpful error
            raise ImportError(f"Could not import {func_name} from {new_module_name}")
    
except ImportError:
    # Module not available, will use the new location
    pass