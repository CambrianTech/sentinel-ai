# File Movement Plan

This document outlines exactly which files need to be moved from their current locations to the new structure. Each section represents a different component or module.

## Core Entry Points

| Current Path | New Path |
|-------------|----------|
| `/main.py` | `/scripts/entry_points/inference.py` |
| `/train.py` | `/scripts/entry_points/train.py` |
| `/generate_samples.py` | `/scripts/entry_points/generate.py` |
| `/model_probe.py` | `/scripts/entry_points/probe.py` |

## Models

| Current Path | New Path |
|-------------|----------|
| `/models/adaptive_transformer.py` | `/sentinel/models/adaptive/transformer.py` |
| `/models/agency_specialization.py` | `/sentinel/models/adaptive/agency_specialization.py` |
| `/models/bloom_adapter.py` | `/sentinel/models/adaptive/bloom_adapter.py` |
| `/models/llama_adapter.py` | `/sentinel/models/adaptive/llama_adapter.py` |
| `/models/optimized_attention.py` | `/sentinel/models/utils/optimized_attention.py` |
| `/models/specialization_registry.py` | `/sentinel/models/utils/specialization_registry.py` |
| `/models/unet_transformer.py` | `/sentinel/models/adaptive/unet_transformer.py` |
| `/models/unet_transformer_optimized.py` | `/sentinel/models/adaptive/unet_transformer_optimized.py` |
| `/models/loaders/*.py` | `/sentinel/models/loaders/` (preserve filenames) |
| `/models/optimized/*.py` | `/sentinel/models/optimized/` (preserve filenames) |

## Controllers

| Current Path | New Path |
|-------------|----------|
| `/controller/controller_ann.py` | `/sentinel/controller/controller_ann.py` |
| `/controller/controller_manager.py` | `/sentinel/controller/controller_manager.py` |
| `/controller/metrics/*.py` | `/sentinel/controller/metrics/` (preserve filenames) |
| `/controller/visualizations/*.py` | `/sentinel/controller/visualizations/` (preserve filenames) |

## Pruning and Fine-Tuning

| Current Path | New Path |
|-------------|----------|
| `/utils/pruning/pruning_module.py` | `/sentinel/pruning/pruning_module.py` |
| `/utils/pruning/strategies.py` | `/sentinel/pruning/strategies/base.py` |
| `/utils/pruning/fine_tuner.py` | `/sentinel/pruning/fine_tuning/base.py` |
| `/utils/pruning/fine_tuner_improved.py` | `/sentinel/pruning/fine_tuning/improved.py` |
| `/utils/pruning/fine_tuner_consolidated.py` | `/sentinel/pruning/fine_tuning/consolidated.py` |
| `/utils/pruning/experiment.py` | `/sentinel/pruning/experiment.py` |
| `/utils/pruning/results_manager.py` | `/sentinel/pruning/results_manager.py` |
| `/utils/pruning/benchmark.py` | `/sentinel/pruning/benchmark.py` |
| `/utils/pruning/visualization.py` | `/sentinel/pruning/visualization.py` |
| `/utils/pruning/stability/*.py` | `/sentinel/pruning/stability/` (preserve filenames) |
| `/utils/pruning/growth.py` | `/sentinel/pruning/growth.py` |
| `/utils/pruning/head_lr_manager.py` | `/sentinel/pruning/head_lr_manager.py` |

## Neural Plasticity

| Current Path | New Path |
|-------------|----------|
| `/utils/adaptive/adaptive_plasticity.py` | `/sentinel/plasticity/adaptive/adaptive_plasticity.py` |
| `/utils/adaptive/__init__.py` | `/sentinel/plasticity/adaptive/__init__.py` |

## Data Handling

| Current Path | New Path |
|-------------|----------|
| `/sentinel_data/__init__.py` | `/sentinel/data/__init__.py` |
| `/sentinel_data/dataset_loader.py` | `/sentinel/data/loaders/dataset_loader.py` |
| `/sentinel_data/eval.py` | `/sentinel/data/eval.py` |
| `/custdata/loaders/*.py` | `/sentinel/data/loaders/custom/` (preserve filenames) |

## Utilities

| Current Path | New Path |
|-------------|----------|
| `/utils/metrics.py` | `/sentinel/utils/metrics/base.py` |
| `/utils/metrics_logger.py` | `/sentinel/utils/metrics/logger.py` |
| `/utils/head_metrics.py` | `/sentinel/utils/metrics/head_metrics.py` |
| `/utils/charting.py` | `/sentinel/utils/visualization/charting.py` |
| `/utils/checkpoint.py` | `/sentinel/utils/checkpoints/checkpoint.py` |
| `/utils/progress_tracker.py` | `/sentinel/utils/progress_tracker.py` |
| `/utils/head_lr_manager.py` | `/sentinel/utils/head_lr_manager.py` |
| `/utils/training.py` | `/sentinel/utils/training.py` |
| `/utils/training_stability.py` | `/sentinel/utils/training_stability.py` |
| `/utils/dynamic_architecture.py` | `/sentinel/utils/dynamic_architecture.py` |
| `/utils/generation_wrapper.py` | `/sentinel/utils/generation_wrapper.py` |
| `/utils/model_wrapper.py` | `/sentinel/utils/model_wrapper.py` |
| `/utils/train_utils.py` | `/sentinel/utils/train_utils.py` |
| `/utils/utils.py` | `/sentinel/utils/utils.py` |
| `/utils/colab/*.py` | `/sentinel/utils/colab/` (preserve filenames) |

## Scripts

| Current Path | New Path |
|-------------|----------|
| `/scripts/analyze_heads.py` | `/scripts/analysis/analyze_heads.py` |
| `/scripts/analyze_pruning_results.py` | `/scripts/analysis/analyze_pruning_results.py` |
| `/scripts/benchmark.py` | `/scripts/benchmarks/benchmark.py` |
| `/scripts/benchmark_agency.py` | `/scripts/benchmarks/benchmark_agency.py` |
| `/scripts/benchmark_optimization.py` | `/scripts/benchmarks/benchmark_optimization.py` |
| `/scripts/benchmark_pruning.py` | `/scripts/benchmarks/benchmark_pruning.py` |
| `/scripts/inference.py` | `/scripts/inference/inference.py` |
| `/scripts/inference_with_pruning.py` | `/scripts/inference/inference_with_pruning.py` |
| `/scripts/finetune_pruned_model.py` | `/scripts/fine_tuning/finetune_pruned_model.py` |
| `/scripts/prune_heads.py` | `/scripts/pruning/prune_heads.py` |
| `/scripts/pruning_and_finetuning.py` | `/scripts/pruning/pruning_and_finetuning.py` |
| `/scripts/neural_plasticity_*.py` | `/scripts/plasticity/` (preserve filenames) |
| `/scripts/run_adaptive_plasticity.py` | `/scripts/plasticity/run_adaptive_plasticity.py` |
| `/scripts/profile_*.py` | `/scripts/profiling/` (preserve filenames) |

## Tests

| Current Path | New Path |
|-------------|----------|
| `/scripts/test_*.py` | Either to `/tests/unit/` or `/tests/integration/` based on content |
| `/tests/test_model_support.py` | `/tests/integration/test_model_support.py` |
| `/tests/test_neural_plasticity.py` | `/tests/integration/test_neural_plasticity.py` |
| `/tests/test_optimization_comparison.py` | `/tests/performance/test_optimization_comparison.py` |
| `/tests/utils/test_memory.py` | `/tests/unit/utils/test_memory.py` |
| `/tests/utils/test_nan_prevention.py` | `/tests/unit/utils/test_nan_prevention.py` |
| `/examples/pruning_experiments/tests/*.py` | `/tests/unit/pruning/` (preserve filenames) |

## Experiments

| Current Path | New Path |
|-------------|----------|
| `/output/plasticity_experiments/*` | `/experiments/results/plasticity/` |
| `/profiling_results/*` | `/experiments/results/profiling/` |
| `/pruning_results/*` | `/experiments/results/pruning/` |
| `/validation_results/*` | `/experiments/results/validation/` |
| `/examples/pruning_experiments/scripts/*.py` | `/experiments/scripts/pruning/` |
| `/examples/pruning_experiments/notebooks/*.ipynb` | `/experiments/notebooks/pruning/` |
| `/examples/controller_agency_demo.py` | `/experiments/scripts/controller/controller_agency_demo.py` |
| `/examples/agency_specialization_demo.py` | `/experiments/scripts/agency/agency_specialization_demo.py` |
| `/colab_notebooks/*.ipynb` | `/experiments/notebooks/colab/` |
| `/colab_notebooks/*.py` | `/experiments/scripts/colab/` |

## Documentation

| Current Path | New Path |
|-------------|----------|
| `/docs/*.md` | `/docs/guides/` |
| `/docs/assets/figures/*.md` | `/docs/assets/figures/` |
| `/docs/improved_diagrams/*.md` | `/docs/assets/diagrams/` |
| `/docs/pruning/*.md` | `/docs/guides/pruning/` |
| `/README.md` | Stay in place |
| `/SUMMARY.md` | `/docs/guides/SUMMARY.md` |
| `/CHANGELOG.md` | Stay in place |
| `/CLAUDE.md` | Stay in place |
| `/LICENSE` | Stay in place |

## Code Movement Implementation Strategy

1. **Create stubs for backward compatibility**:
   - For example, when moving `utils/pruning/pruning_module.py` to `sentinel/pruning/pruning_module.py`:
   - Create stub at old location: `utils/pruning/pruning_module.py`
   - Stub content:
     ```python
     """
     DEPRECATED: This module has moved to sentinel.pruning.pruning_module
     This import stub will be removed in a future version.
     """
     import warnings
     warnings.warn(
         "Importing from utils.pruning.pruning_module is deprecated. "
         "Use sentinel.pruning.pruning_module instead.",
         DeprecationWarning, 
         stacklevel=2
     )
     
     from sentinel.pruning.pruning_module import *
     ```

2. **Update imports in moved files**:
   - For example, in `sentinel/pruning/pruning_module.py`, update:
     - `from utils.pruning.strategies import ...` to `from sentinel.pruning.strategies import ...`
     - `from models.loaders import ...` to `from sentinel.models.loaders import ...`

3. **Create the file movement tool**:
   - Implement a tool based on this plan that:
     1. Creates the target directory if needed
     2. Copies the file content to the new location
     3. Updates imports in the copied file
     4. Creates a stub file at the old location
     5. Logs the movement for later reference

4. **Movement verification tests**:
   - Create a test for each moved file that:
     1. Verifies both old and new imports work
     2. Verifies behavior is identical
     3. Reports any issues or inconsistencies

## Usage Examples Update

After reorganizing, update usage examples in documentation:

### Old Usage

```python
from utils.pruning.pruning_module import PruningModule
from utils.pruning.strategies import get_strategy

module = PruningModule("distilgpt2")
strategy = get_strategy("entropy", module)
```

### New Usage

```python
from sentinel.pruning import PruningModule
from sentinel.pruning.strategies import get_strategy

module = PruningModule("distilgpt2")
strategy = get_strategy("entropy", module)
```

## Entry Point Updates

Replace:
```bash
python main.py --model distilgpt2 --prompt "Hello, world"
```

With:
```bash
python -m sentinel inference --model distilgpt2 --prompt "Hello, world"
```

Or after installation:
```bash
sentinel-inference --model distilgpt2 --prompt "Hello, world"
```