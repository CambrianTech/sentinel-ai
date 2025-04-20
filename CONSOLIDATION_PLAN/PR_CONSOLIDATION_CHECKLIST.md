# PR Consolidation Checklist

This file tracks our consolidation plan for the neural plasticity feature PR. The goal is to reduce complexity and redundancy by organizing the implementation into a clean modular structure.

## Core Implementation

- [x] scripts/neural_plasticity/run_experiment.py - **KEEP**: Main consolidated implementation
- [x] scripts/neural_plasticity/examples/minimal_example.py - **KEEP**: Minimal example for users
- [x] scripts/neural_plasticity/full_experiment.py - **KEEP**: Full experiment implementation
- [x] scripts/neural_plasticity/simple_experiment.py - **KEEP**: Simplified experiment implementation
- [x] scripts/neural_plasticity/visualization/dashboard_generator.py - **KEEP**: Dashboard generation
- [x] scripts/run_neural_plasticity.py - **KEEP**: Main entry point that calls into the modular implementation

## `sentinel` Package Files

- [x] sentinel/pruning/plasticity_controller.py - **KEEP**: Core controller implementation
- [x] sentinel/pruning/dual_mode_pruning.py - **KEEP**: Dual mode pruning implementation
- [x] sentinel/pruning/experiment_runner.py - **KEEP**: Experiment runner
- [x] sentinel/pruning/model_manager.py - **KEEP**: Model management
- [x] sentinel/pruning/gradient_based_pruning_controller.py - **KEEP**: Gradient-based controller
- [x] sentinel/pruning/visualization/visualization.py - **KEEP**: Visualization utilities
- [x] sentinel/pruning/test_dual_mode_pruning.py - **KEEP**: Test for dual mode pruning
- [x] sentinel/pruning/text_generator.py - **KEEP**: Text generation utilities
- [x] sentinel/upgrayedd/* - **KEEP**: Upgrayedd implementation

## Tests

- [x] tests/unit/plasticity/test_neural_plasticity_modular.py - **KEEP**: New tests for modular implementation
- [x] tests/unit/plasticity/test_neural_plasticity.py - **KEEP**: Tests for neural plasticity
- [x] tests/sentinel/pruning/test_plasticity_controller.py - **KEEP**: Tests for plasticity controller
- [x] tests/unit/utils/test_neural_plasticity.py - **KEEP**: Tests for neural plasticity utilities
- [x] tests/sentinel/pruning/integration_test_plasticity.py - **KEEP**: Integration tests

## Utils Implementation

- [x] utils/neural_plasticity/* - **KEEP**: Utility implementation for neural plasticity
- [x] utils/adaptive/adaptive_plasticity.py - **KEEP**: Adaptive plasticity implementation
- [x] utils/pruning/api/* - **KEEP**: API for pruning functionality

## Documentation

- [x] NEURAL_PLASTICITY_README.md - **KEEP**: Main documentation
- [x] NEURAL_PLASTICITY_ROADMAP.md - **KEEP**: Roadmap for future work
- [x] scripts/NEURAL_PLASTICITY_EXPERIMENT.md - **KEEP**: Experiment documentation
- [x] scripts/README_NEURAL_PLASTICITY.md - **KEEP**: Additional documentation
- [x] utils/neural_plasticity/README.md - **KEEP**: Utility documentation

## Colab Notebooks

- [x] colab_notebooks/NeuralPlasticityDemo.ipynb - **KEEP**: Main demo notebook
- [x] colab_notebooks/PlasticityIntegration.py - **KEEP**: Integration utilities
- [x] neural_plasticity_minimal_test.ipynb - **KEEP**: Minimal test notebook
- [x] neural_plasticity_runnable.ipynb - **KEEP**: Runnable notebook
- [x] neural_plasticity_test_executed.ipynb - **KEEP**: Executed test notebook
- [x] neural_plasticity_ultra_minimal.ipynb - **KEEP**: Ultra minimal notebook

## Notebook Utilities

- [x] colab_notebooks/neural_plasticity_dashboard_cell.py - **KEEP**: Dashboard cell for colab
- [x] utils/colab/neural_plasticity_dashboard_cell.py - **KEEP**: Dashboard cell utility
- [x] notebook_maintenance/fix_colab_compatibility.py - **KEEP**: Fix for colab compatibility
- [x] notebook_maintenance/comprehensive_notebook_validation.py - **KEEP**: Comprehensive validation

## Fix and Clean Up

- [ ] temp_fix/* - **REMOVE**: These should all be consolidated or removed
- [ ] fix_dataset_imports.py - **REMOVE**: Redundant
- [ ] fix_notebook.py - **REMOVE**: Redundant
- [ ] increment_notebook_version.py - **CONSIDER**: May be useful for notebook versioning

## Example Files

- [x] examples/neural_plasticity_example.py - **KEEP**: Example implementation

## Test Files

- [ ] minimal_pruning_test.py - **REMOVE**: Redundant with test in scripts/neural_plasticity
- [ ] test_dataset_loading.py - **REMOVE**: Should be part of proper test suite
- [ ] test_run_experiment.py - **REMOVE**: Redundant with proper tests

## Scripts

- [ ] run_experiment.py - **REMOVE**: Redundant with scripts/neural_plasticity/run_experiment.py
- [ ] scripts/neural_plasticity_cycle.py - **REMOVE**: Redundant with modular implementation
- [ ] scripts/neural_plasticity_demo.py - **REMOVE**: Redundant with modular implementation
- [ ] scripts/display_warmup_in_colab.py - **CONSIDER**: Might be useful for colab integration
- [ ] scripts/generate_comprehensive_results.py - **CONSIDER**: Might be useful for results generation
- [ ] scripts/run_notebook_minimal.py - **REMOVE**: Redundant
- [ ] scripts/test_blas_stability.py - **KEEP**: Important for stability testing
- [ ] scripts/test_gpu_compatibility.py - **KEEP**: Important for GPU compatibility
- [ ] scripts/test_gradient_pruning.py - **KEEP**: Important test
- [ ] scripts/test_pruning_algorithm.py - **KEEP**: Important test
- [ ] scripts/test_tensor_only.py - **KEEP**: Important test

## Visualization Tests

- [ ] scripts/viz_tests/* - **KEEP**: Important visualization tests

## Next Steps

1. Remove all temp_fix/* files as they should be consolidated
2. Remove redundant scripts at repository root
3. Remove redundant test files that are superseded by the modular implementation
4. Ensure that all tests still pass with the consolidated structure
5. Update documentation to reflect the new organization
6. Create a minimal end-to-end test that uses the consolidated implementation

## Open Questions

1. Should we move all visualization code to scripts/neural_plasticity/visualization/?
2. Should we consolidate all notebook-related utilities to a single location?
3. Are there any interdependencies we need to be careful about when removing files?