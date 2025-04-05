# sentinel_data/table.py
"""
Compatibility module for HuggingFace datasets library.

This module provides compatibility functions to resolve import conflicts between 
the HuggingFace datasets library and our project's sdata module.
"""

def table_cast(table, *args, **kwargs):
    """
    Compatibility function for HuggingFace datasets.
    
    This is a stub implementation to avoid import errors in the HuggingFace datasets 
    library. It should never be called in practice as we're using our own renamed import.
    
    Args:
        table: Table to cast
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        The input table unchanged
    """
    raise NotImplementedError(
        "This is a stub implementation for compatibility. "
        "It should not be called directly. Use the sdata module instead."
    )