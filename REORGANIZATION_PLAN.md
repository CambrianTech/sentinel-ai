# Repository Reorganization Plan

## Current Understanding

The repository contains several key components:

1. **Model Architecture**
   - Adaptive transformer models in `models/`
   - Agency specialization in `models/agency_specialization.py`
   - Model loaders in `models/loaders/`
   - Optimized versions in `models/optimized/`

2. **Controllers**
   - Controller logic in `controller/`
   - Metrics and visualizations in `controller/metrics/` and `controller/visualizations/`

3. **Pruning & Fine-tuning**
   - Pruning utilities in `utils/pruning/`
   - Fine-tuning logic in various locations

4. **Neural Plasticity**
   - Adaptive plasticity in `utils/adaptive/`
   - Adaptive experiments in `scripts/`

5. **Data Handling**
   - Dataset loading in `sentinel_data/`
   - Custom data loaders in `custdata/loaders/`

6. **Outputs & Results**
   - Multiple result directories: `output/`, `pruning_results/`, `profiling_results/`, etc.
   - Validation in `validation_results/`
   - Benchmark data across various locations

## Issues to Address

1. **Scattered Entry Points**
   - Multiple scripts generating similar outputs in different locations
   - Unclear relationship between entry points and output directories

2. **Inconsistent Output Structure**
   - Results scattered across many directories
   - Similar experiments producing outputs in different locations

3. **Test Coverage Gaps**
   - Many tests are in `scripts/` rather than a dedicated test directory
   - Lack of systematic unit testing for core modules

4. **Unclear Data Flow**
   - Multiple data loading paths (`sentinel_data/`, `custdata/`, `datasets/`)
   - Duplication of data handling logic

## Reorganization Strategy

### 1. Directory Structure

```
sentinel-ai/
├── sentinel/                   # Main package directory
│   ├── __init__.py             # Package initialization
│   ├── models/                 # Model definitions
│   │   ├── __init__.py
│   │   ├── adaptive/           # Adaptive transformer models
│   │   ├── utils/              # Model utility functions
│   │   └── loaders/            # Model loading utilities
│   ├── controller/             # Controller logic
│   │   ├── __init__.py
│   │   ├── metrics/            # Metrics collection
│   │   └── visualizations/     # Visualization utilities
│   ├── pruning/                # Pruning functionality
│   │   ├── __init__.py
│   │   ├── strategies/         # Pruning strategies
│   │   ├── fine_tuning/        # Fine-tuning after pruning
│   │   └── stability/          # Stability enhancements
│   ├── plasticity/             # Neural plasticity system
│   │   ├── __init__.py
│   │   ├── adaptive/           # Adaptive system
│   │   └── metrics/            # Plasticity metrics
│   ├── data/                   # Data handling
│   │   ├── __init__.py
│   │   ├── loaders/            # Dataset loaders
│   │   └── processors/         # Data processors
│   └── utils/                  # General utilities
│       ├── __init__.py
│       ├── metrics/            # General metrics
│       ├── visualization/      # Visualization tools
│       └── checkpoints/        # Checkpoint handling
├── experiments/                # All experiments
│   ├── configs/                # Experiment configurations
│   ├── scripts/                # Experiment scripts
│   ├── notebooks/              # Experiment notebooks
│   └── results/                # Structured results directory
│       ├── pruning/            # Pruning experiment results
│       ├── plasticity/         # Plasticity experiment results
│       ├── profiling/          # Performance profiling results
│       └── validation/         # Validation experiment results
├── tests/                      # All tests
│   ├── unit/                   # Unit tests for all modules
│   ├── integration/            # Integration tests
│   ├── performance/            # Performance tests
│   └── fixtures/               # Test fixtures
├── scripts/                    # Entry point scripts
│   ├── train.py                # Training script
│   ├── inference.py            # Inference script
│   ├── prune.py                # Pruning script
│   └── benchmark.py            # Benchmarking script
├── legacy/                     # Legacy code (retained for reference)
├── docs/                       # Documentation
│   ├── api/                    # API documentation
│   ├── examples/               # Usage examples
│   └── guides/                 # User guides
├── setup.py                    # Package setup
└── requirements.txt            # Package requirements
```

### 2. Restructuring Strategies

#### Centralize Entry Points

1. Create consistent entry point scripts in `scripts/` with:
   - Standardized argument parsing
   - Clear output paths
   - Comprehensive help documentation

2. Each entry point should handle one type of task:
   - `train.py` - Training models
   - `inference.py` - Running inference
   - `prune.py` - Pruning models
   - `benchmark.py` - Benchmarking performance

#### Consolidate Results

1. Move all results under the `experiments/results/` directory with subdirectories:
   - `experiments/results/pruning/` for pruning results
   - `experiments/results/plasticity/` for plasticity experiment results
   - `experiments/results/profiling/` for profiling results
   - `experiments/results/validation/` for validation results

2. Update all scripts to use these standardized output locations

#### Improve Test Coverage

1. Move all test scripts from `scripts/test_*.py` to appropriate test directories:
   - Unit tests in `tests/unit/`
   - Integration tests in `tests/integration/`
   - Performance tests in `tests/performance/`

2. Create unit tests for all key modules that currently lack them

#### Unify Data Handling

1. Consolidate all data handling under `sentinel/data/`
   - Move `sentinel_data/` functionality to `sentinel/data/`
   - Move `custdata/` functionality to `sentinel/data/`
   - Keep a standardized interface for all dataset loading

### 3. Migration Plan

#### Phase 1: Core Structure

1. Create the new directory structure
2. Move core modules to their new locations
3. Create proper `__init__.py` files to maintain imports
4. Add compatibility imports for backward compatibility

#### Phase 2: Entry Points

1. Create standardized entry points in `scripts/`
2. Update entry points to use new module structure
3. Create script redirects for backward compatibility

#### Phase 3: Test Migration

1. Move all test files to their proper locations
2. Update tests to use new module structure
3. Expand test coverage for all core modules

#### Phase 4: Data and Configuration

1. Consolidate data handling
2. Update configuration handling
3. Standardize all path references

#### Phase 5: Result Migration

1. Create standard result directory structure
2. Update all output paths in scripts
3. Add documentation about result organization

### 4. Detailed Mapping of Current to New Paths

| Current Path | New Path | Notes |
|---|---|---|
| `main.py` | `scripts/inference.py` | Main inference entry point |
| `train.py` | `scripts/train.py` | Main training entry point |
| `models/` | `sentinel/models/` | All model definitions |
| `models/adaptive_transformer.py` | `sentinel/models/adaptive/transformer.py` | Adaptive transformer model |
| `models/loaders/` | `sentinel/models/loaders/` | Model loading utilities |
| `models/optimized/` | `sentinel/models/optimized/` | Optimized model implementations |
| `controller/` | `sentinel/controller/` | Controller logic |
| `utils/pruning/` | `sentinel/pruning/` | Pruning functionality |
| `utils/adaptive/` | `sentinel/plasticity/adaptive/` | Adaptive plasticity system |
| `utils/pruning/fine_tuner*.py` | `sentinel/pruning/fine_tuning/` | Fine-tuning utilities |
| `utils/pruning/stability/` | `sentinel/pruning/stability/` | Stability enhancements |
| `sentinel_data/` | `sentinel/data/` | Data handling |
| `custdata/` | `sentinel/data/custom/` | Custom data handling |
| `scripts/test_*.py` | `tests/unit/` or `tests/integration/` | Test scripts |
| `output/` | `experiments/results/` | Result directory |
| `pruning_results/` | `experiments/results/pruning/` | Pruning results |
| `profiling_results/` | `experiments/results/profiling/` | Profiling results |
| `validation_results/` | `experiments/results/validation/` | Validation results |

### 5. Implementation Details

#### File Operations

For each file move:
1. Create target directory if it doesn't exist
2. Copy file to new location
3. Update import statements in the file
4. Add compatibility imports in old location (temporarily)
5. Update references to the file in other modules

#### Import Compatibility

To maintain backward compatibility:
1. Create stub files at old locations that import from new locations
2. Add deprecation warnings to encourage updates to new imports
3. Document new import patterns in README and docs

#### Path References

For all hardcoded paths:
1. Replace with centralized path configuration
2. Use relative imports where possible
3. Add path utilities for finding data and output directories

#### Test Updates

For all tests:
1. Update import statements to use new module paths
2. Update any hardcoded paths to use centralized path configuration
3. Add fixtures for commonly used test setups

## Timeline and Priorities

### Phase 1: Core Reorganization (1-2 days)
- Create directory structure 
- Move core modules
- Set up compatibility imports

### Phase 2: Entry Points & Tests (1-2 days)
- Create standardized entry points
- Move tests to proper locations
- Add initial unit tests for uncovered modules

### Phase 3: Data & Results (2-3 days)
- Consolidate data handling
- Standardize result directories
- Update all scripts to use new paths

### Phase 4: Documentation & Testing (2-3 days)
- Update documentation to reflect new structure
- Add comprehensive tests for all key functionality
- Verify all entry points work correctly

### Phase 5: Cleanup & Finalization (1 day)
- Remove compatibility imports
- Clean up any remaining legacy code
- Final testing of all functionality

## Total Estimated Time: 7-11 days

## Next Steps

1. Create the basic directory structure
2. Start moving core modules
3. Set up continuous integration for testing
4. Incrementally update and test each component