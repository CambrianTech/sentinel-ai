# SentinelAI Tests

This directory contains test files for the SentinelAI project that verify core functionality.

## Test Files

- `test_model_support.py`: Tests compatibility with various model architectures (GPT-2, OPT, Bloom, etc.)
- `test_neural_plasticity.py`: Tests the adaptive neural plasticity system
- `test_optimization_comparison.py`: Tests optimization performance across different strategies

## Test Scripts

In addition to these core test files, we have extensive test scripts in the `scripts/` directory:

- `test_modular_experiment.py`: Tests the modular experiment framework
- `test_improved_fine_tuner.py`: Tests the improved fine-tuner with stability enhancements
- `test_consolidated_fine_tuner.py`: Tests the consolidated fine-tuner with performance improvements
- `test_head_growth.py` and `test_head_growth_unit.py`: Tests head growth functionality
- `test_adaptive_plasticity.py`: Tests the adaptive plasticity system
- `test_optimized_model.py` and `test_optimized_performance.py`: Tests model optimization

## Running Tests

### Basic Tests

```bash
# Run model support tests
python -m tests.test_model_support --device cpu

# Run neural plasticity tests
python -m tests.test_neural_plasticity --model distilgpt2

# Run optimization comparison tests
python -m tests.test_optimization_comparison
```

### Experiment Tests

```bash
# Test the modular experiment framework
python scripts/test_modular_experiment.py --model distilgpt2 --strategy entropy --pruning_level 0.3

# Test the improved fine-tuner
python scripts/test_improved_fine_tuner.py --model distilgpt2 --strategy entropy --pruning_level 0.3 --epochs 2
```

## Test Coverage

The test suite covers the following key functionalities:

1. **Model Support**: Testing for compatibility with different model architectures
2. **Pruning**: Testing for different pruning strategies and levels
3. **Fine-tuning**: Testing post-pruning fine-tuning with various configurations
4. **Adaptive Plasticity**: Testing the neural plasticity system for adaptive optimization
5. **Stability**: Testing the NaN prevention and memory management systems
6. **Performance**: Testing optimization to improve inference and training speed

## Adding New Tests

When adding new features, please add corresponding tests that verify:

1. Basic functionality works as expected
2. Edge cases are handled correctly 
3. Integration with existing components works properly
4. Performance characteristics are acceptable