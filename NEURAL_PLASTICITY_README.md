# Neural Plasticity on Apple Silicon

## Core Features and Optimizations for Apple Silicon

This implementation of Neural Plasticity has been optimized to work reliably on Apple Silicon (M1/M2/M3) while maintaining full GPU support on Colab. The key features include:

### 1. Environment Detection

- Automatically detects Apple Silicon vs. Colab environments
- Configures optimizations based on the specific hardware
- Seamlessly adapts tensor operations for best performance on each platform

### 2. Apple Silicon Optimizations

- Prevents BLAS/libtorch crashes on Apple Silicon with multi-level fallback:
  - Level 1: Uses NumPy-based matrix multiplication when possible
  - Level 2: Uses manual Python implementation for smaller matrices
  - Level 3: Uses protected single-threaded PyTorch operations
- Forces single-threaded BLAS operations
- Disables problematic optimizations that cause crashes
- Handles memory layout and contiguity issues automatically
- Auto-handles tensor devices and gradients

### 3. Colab GPU Support

- Automatically detects and utilizes GPU in Colab
- Places tensors on GPU for maximum speed
- Handles device transitions seamlessly
- Optimizes batch sizes based on available GPU memory

### 4. Universal Compatibility

- Works consistently across all environments:
  - Apple Silicon (M1/M2/M3)
  - Colab with GPU
  - Colab without GPU
  - Standard x86 CPUs
- No code changes needed when moving between environments

### 5. Robust Error Handling

- Proper handling of NaN/Inf values
- Graceful degradation when operations fail
- Detailed error reporting for debugging
- Auto-recovery from numerical issues

## Key Components

1. **safe_matmul**: A robust matrix multiplication function that works reliably on all platforms
2. **calculate_head_entropy**: Entropy calculation for attention heads with proper numerical stability
3. **IS_APPLE_SILICON**: Detection flag for Apple Silicon optimization
4. **Visualization enhancements**: Proper tensor handling for visualization on all platforms

## Testing

All components have been thoroughly tested on Apple Silicon with comprehensive tests:

1. **Matrix Stability Test**: Tests matrix multiplication with various sizes (small to large)
2. **Entropy Calculation Test**: Tests entropy calculation on attention maps
3. **Large Tensor Test**: Tests handling of very large tensors (1000x1000 and larger)
4. **Visualization Test**: Tests visualization capabilities with proper tensor handling

## Usage

Simply use the functions provided by the Neural Plasticity module. The environment detection and optimization happens automatically.

```python
from utils.neural_plasticity.core import safe_matmul, calculate_head_entropy

# Matrix multiplication (works on all platforms)
result = safe_matmul(matrix_a, matrix_b)

# Entropy calculation (works on all platforms)
entropy = calculate_head_entropy(attention_maps)
```

## Running the NeuralPlasticityDemo Notebook

To run the NeuralPlasticityDemo notebook safely on Apple Silicon:

1. Use the provided runner script:
   ```bash
   python run_neural_plasticity_safe.py
   ```

2. Or manually run the verified test:
   ```bash
   python neural_plasticity_test_verified.py
   ```

The implementation successfully passes all tests on Apple Silicon, ensuring reliable operation without crashes.

## Version History

- v0.0.58 (2025-04-19): Added comprehensive Apple Silicon support with multi-level fallback and cross-platform compatibility