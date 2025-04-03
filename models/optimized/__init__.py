"""
Optimized model implementations with different optimization strategies.

This package contains highly optimized implementations of the transformer model
that address different aspects of performance bottlenecks identified through
profiling.

Optimization Levels:
- Level 0: No optimizations (debugging only)
- Level 1: Default optimizations (balanced)
- Level 2: Aggressive optimizations (best for CPU)
- Level 3: Extreme optimizations (specialized for GPU)

The default optimization level is now set to 2 based on comprehensive 
profiling that showed it provides the best balance of performance and 
functionality on CPU devices.
"""

# Set default optimization level based on profiling results
import os
if "OPTIMIZATION_LEVEL" not in os.environ:
    # Level 2 showed the best performance on CPU in our profiling
    os.environ["OPTIMIZATION_LEVEL"] = "2"