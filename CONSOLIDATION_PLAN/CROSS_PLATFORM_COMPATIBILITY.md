# Cross-Platform Compatibility Plan

This document outlines the plan to ensure the neural plasticity implementation works seamlessly across different environments: macOS (especially Apple Silicon), Ubuntu GPU servers, and Google Colab (CPU/GPU).

## Environment-Specific Challenges

### 1. Apple Silicon (M1/M2/M3)
- **Issue**: BLAS/libtorch crashes due to compatibility issues with MPS
- **Solution**: 
  - Detect Apple Silicon hardware at runtime
  - Move tensors to CPU when performing operations known to crash
  - Use platform-specific optimizations for visualization
  - Override default tensor handling behavior

### 2. Ubuntu GPU
- **Issue**: Memory management and CUDA compatibility
- **Solution**:
  - Ensure proper GPU memory cleanup
  - Add safeguards for out-of-memory errors
  - Implement graceful fallback to CPU when needed
  - Optimize batch sizes for GPU training

### 3. Google Colab
- **Issue**: Environment constraints and notebook integration
- **Solution**:
  - Create Colab-specific integration utilities
  - Implement visualization that works within notebook cells
  - Handle T4 GPU acceleration properly
  - Manage session timeouts gracefully

## Implementation Strategy

### 1. Environment Detection

```python
def detect_environment():
    """Detect the current execution environment."""
    import platform
    import os
    
    # Check for Colab
    is_colab = 'COLAB_GPU' in os.environ or 'google.colab' in str(get_ipython())
    
    # Check for Apple Silicon
    is_apple_silicon = platform.system() == 'Darwin' and platform.processor() == 'arm'
    
    # Check for GPU
    has_gpu = torch.cuda.is_available()
    
    return {
        'is_colab': is_colab,
        'is_apple_silicon': is_apple_silicon,
        'has_gpu': has_gpu,
        'gpu_type': torch.cuda.get_device_name(0) if has_gpu else None
    }
```

### 2. Tensor Handling Adapters

```python
def safe_tensor_operation(tensor_op_func):
    """Decorator for making tensor operations safe across platforms."""
    @functools.wraps(tensor_op_func)
    def wrapper(tensor, *args, **kwargs):
        env = detect_environment()
        
        # Move tensor to CPU for Apple Silicon when needed
        if env['is_apple_silicon'] and tensor.device.type != 'cpu':
            tensor = tensor.detach().cpu()
            
        # Run the operation
        result = tensor_op_func(tensor, *args, **kwargs)
        
        # Return to original device if needed
        if 'return_device' in kwargs and kwargs['return_device'] is not None:
            result = result.to(kwargs['return_device'])
            
        return result
    return wrapper
```

### 3. Visualization Adapters

```python
def initialize_visualization_backend():
    """Initialize the visualization backend based on environment."""
    env = detect_environment()
    
    if env['is_colab']:
        # Use inline backend for Colab
        import matplotlib
        matplotlib.use('inline')
        from google.colab import output
        enable_interactive_plots()
    elif env['is_apple_silicon']:
        # Use Agg backend for Apple Silicon to avoid crashes
        import matplotlib
        matplotlib.use('Agg')
    else:
        # Standard backend
        import matplotlib
        matplotlib.use('TkAgg')
```

### 4. Memory Management

```python
def manage_gpu_memory(func):
    """Decorator for managing GPU memory during operations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        env = detect_environment()
        
        try:
            # Run the operation
            result = func(*args, **kwargs)
            return result
        finally:
            # Clean up GPU memory if needed
            if env['has_gpu']:
                import torch
                import gc
                torch.cuda.empty_cache()
                gc.collect()
    return wrapper
```

## Integration with Core Components

1. **Modify `scripts/neural_plasticity/run_experiment.py`**:
   - Add environment detection at startup
   - Configure tensor handling based on environment
   - Set up visualization backend appropriately

2. **Modify `sentinel/plasticity/plasticity_loop.py`**:
   - Add memory management for GPU operations
   - Implement safe tensor handling for Apple Silicon
   - Add graceful fallbacks for all platforms

3. **Create `scripts/neural_plasticity/colab/integration.py`**:
   - Add Colab-specific integration utilities
   - Implement notebook cell visualization helpers
   - Create progress tracking for Colab

## Testing Strategy

1. **Create Environment-Specific Test Suite**:
   - Test on Apple Silicon (M1/M2/M3)
   - Test on Ubuntu with NVIDIA GPU
   - Test in Colab with CPU and T4 GPU

2. **Verify Cross-Platform Feature Coverage**:
   - Verify all core features work on all platforms
   - Confirm visualization works in all environments
   - Validate dashboard generation across platforms

3. **Implement CI Tests**:
   - Add CI tests for different environments
   - Create environment-specific test fixtures
   - Add runtime environment detection tests

## Expected Outcomes

1. **Unified Interface**:
   - Single API that works across all environments
   - No environment-specific code in core functionality
   - Automatic adaptation to the current platform

2. **Optimal Performance**:
   - GPU acceleration when available
   - Safeguards for Apple Silicon
   - Efficient memory usage on all platforms

3. **Consistent Visualization**:
   - Same visualizations on all platforms
   - Colab integration for notebook cells
   - HTML dashboard generation everywhere

4. **Robust Error Handling**:
   - Graceful fallbacks for all operations
   - Helpful error messages for environment-specific issues
   - Automatic recovery from common failures