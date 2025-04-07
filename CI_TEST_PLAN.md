# Continuous Integration Test Plan

This document outlines a comprehensive testing strategy for the Sentinel AI codebase, with the goal of achieving close to 100% test coverage for our CI pipeline.

## Current Test Coverage

### ✅ Tested Components
1. **AdaptiveOptimizerConfig** class - `sentinel/upgrayedd/tests/test_config.py`
2. **Dataset Loading with Mock** - `test_dataset_loading.py` 
3. **Experiment Runner Integration** - `test_run_experiment.py`
4. **Notebook Integration** - Tested via PruningAndFineTuningColab.py super_simple mode

### ❌ Components Needing Tests

#### Core Modules
1. **AdaptiveOptimizer** (sentinel/upgrayedd/optimizer/adaptive_optimizer.py)
   - Need tests for all methods: prune_model, fine_tune, evaluate, save_checkpoint, etc.
   - Currently only tested through integration tests

2. **Pruning Strategies** (sentinel/upgrayedd/strategies/)
   - Need unit tests for entropy.py, magnitude.py, random.py
   - Tests should verify head selection logic works correctly

3. **Metrics Tracker** (sentinel/upgrayedd/metrics/tracker.py)
   - Need tests for tracking functions, plotting, and serialization

4. **Utils** (sentinel/upgrayedd/utils/)
   - Need tests for model_utils.py
   - Need tests for training.py (fine-tuning functions)
   - Need tests for generation.py

#### Older API Components
1. **Pruning API** (sentinel/pruning/)
   - Need unit tests for experiment_runner.py
   - Need tests for fixed_pruning_module.py and fixed_pruning_module_jax.py
   - Need tests for model_manager.py and text_generator.py

## Test Plan

### 1. Unit Tests (Priority: High)
For each module, we need to create dedicated unit tests that verify the component works in isolation.

#### AdaptiveOptimizer Tests
```python
# test_adaptive_optimizer.py
def test_prune_model():
    # Test the pruning functionality with a small model
    
def test_fine_tune():
    # Test fine-tuning with minimal training data
    
def test_evaluate():
    # Test evaluation functionality
    
def test_save_and_load_checkpoint():
    # Test checkpointing works correctly
```

#### Strategy Tests
```python
# test_strategies.py
def test_entropy_strategy():
    # Test entropy-based head selection
    
def test_magnitude_strategy():
    # Test magnitude-based head selection
    
def test_random_strategy():
    # Test random head selection
```

#### Utils Tests
```python
# test_model_utils.py
def test_load_model_and_tokenizer():
    # Test model loading functionality
    
def test_prepare_attention_masks():
    # Test attention mask creation for various tokenizers
```

### 2. Integration Tests (Priority: Medium)
Create tests that verify multiple components work together correctly.

```python
# test_pruning_and_fine_tuning.py
def test_prune_and_fine_tune_cycle():
    # Test complete pruning and fine-tuning cycle
    
def test_continuous_optimization():
    # Test multiple optimization cycles
```

### 3. Cross-Module Tests (Priority: Low)
Tests that verify compatibility between different parts of the codebase.

```python
# test_api_compatibility.py
def test_upgrayedd_and_pruning_compatibility():
    # Test that the new and old API can interoperate
```

## Test Implementation Approach

### Phase 1: Critical Unit Tests
1. Create test_adaptive_optimizer.py for comprehensive testing of the optimizer
2. Create test_strategies.py to test all pruning strategies
3. Create test_data_utils.py to ensure robust data loading

### Phase 2: Integration Tests
1. Enhance test_run_experiment.py to test more scenarios
2. Create tests for full pruning and fine-tuning cycles with different models

### Phase 3: Comprehensive Test Suite
1. Add tests for edge cases and error conditions
2. Add performance regression tests
3. Implement test fixtures for reusable test components

## CI Integration Plan

1. **GitHub Actions Setup**
   - Create a CI workflow that runs on each PR
   - Run unit tests and critical integration tests

2. **Test Matrix**
   - Test across multiple Python versions (3.8, 3.9, 3.10)
   - Test with different torch versions

3. **Coverage Reporting**
   - Add coverage measurement (pytest-cov)
   - Set up coverage thresholds for CI passing

## Test Commands

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/unit/test_adaptive_optimizer.py

# Run with coverage
python -m pytest --cov=sentinel tests/
```

## Next Steps

1. Create test directory structure:
```
tests/
├── unit/
│   ├── upgrayedd/
│   │   ├── test_optimizer.py
│   │   ├── test_strategies.py
│   │   └── test_utils.py
│   └── pruning/
│       ├── test_experiment_runner.py
│       └── test_strategies.py
├── integration/
│   ├── test_pruning_cycle.py
│   └── test_notebook_execution.py
└── conftest.py
```

2. Implement highest priority tests first
3. Set up GitHub Actions CI workflow

## Conclusion

By implementing this test plan, we will achieve close to 100% coverage of the critical components in our codebase. This will ensure that our modular pruning API is robust, reliable, and ready for production use.

🤖 Generated with [Claude Code](https://claude.ai/code)