# Code Organization Refactoring Plan

This document outlines the plan for reorganizing the codebase to improve structure, maintainability, and testability.

## Current Issues

1. Duplicate/parallel structures
   - `/sentinel` package contains a partial implementation of a cleaner architecture
   - Many modules are scattered across the codebase without a clear organization
   - New neural plasticity functionality needs proper integration

2. Mixed responsibilities
   - Some modules handle multiple concerns
   - Utilities are mixed with core functionality

3. Inconsistent import patterns
   - Some imports use absolute paths, others use relative
   - Path manipulation in scripts instead of proper package structure

## Target Structure

```
sentinel/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── adaptive/
│   ├── loaders/
│   └── optimized/
├── controller/
│   ├── __init__.py
│   ├── metrics/
│   └── visualizations/
├── pruning/
│   ├── __init__.py
│   ├── fixed_pruning_module.py
│   ├── fixed_pruning_module_jax.py
│   ├── pruning_module.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── entropy.py
│   │   ├── magnitude.py
│   │   └── random.py
│   ├── growth/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── utils.py
│   │   ├── gradient_sensitivity.py
│   │   ├── entropy_gap.py
│   │   ├── balanced.py
│   │   └── random.py
│   ├── fine_tuning/
│   │   ├── __init__.py
│   │   ├── fine_tuner.py
│   │   ├── fine_tuner_consolidated.py
│   │   └── fine_tuner_improved.py
│   ├── stability/
│   │   ├── __init__.py
│   │   ├── memory_management.py
│   │   └── nan_prevention.py
│   └── visualization/
│       ├── __init__.py
│       └── visualization.py
├── data/
│   ├── __init__.py
│   └── loaders/
└── utils/
    ├── __init__.py
    ├── metrics.py
    ├── progress_tracker.py
    └── checkpoints/
```

## Migration Plan

### Phase 1: Pruning Module Migration

1. Move core pruning modules to sentinel/pruning:
   - `utils/pruning/fixed_pruning_module.py` → `sentinel/pruning/fixed_pruning_module.py`
   - `utils/pruning/fixed_pruning_module_jax.py` → `sentinel/pruning/fixed_pruning_module_jax.py`
   - `utils/pruning/pruning_module.py` → `sentinel/pruning/pruning_module.py`

2. Split strategies module:
   - `utils/pruning/strategies.py` → Split into separate files in `sentinel/pruning/strategies/`

3. Split growth module:
   - `utils/pruning/growth.py` → Split into separate files in `sentinel/pruning/growth/`

4. Move fine tuners:
   - `utils/pruning/fine_tuner*.py` → `sentinel/pruning/fine_tuning/`

5. Move stability-related utilities:
   - `utils/pruning/stability/*` → `sentinel/pruning/stability/`

6. Move visualization utilities:
   - `utils/pruning/visualization.py` → `sentinel/pruning/visualization/visualization.py`

### Phase 2: Update Import Statements

1. Update imports in moved files to use the new paths

2. Create compatibility imports in old locations that import from new locations 
   to maintain backward compatibility

3. Update scripts to use new import paths

### Phase 3: Migration Testing

1. Run comprehensive test suite to ensure functionality is maintained

2. Test all scripts that depend on these modules

3. Fix any import or functionality issues

### Phase 4: Documentation Update

1. Update READMEs to reflect new structure

2. Document migration path for existing users

## Backward Compatibility

To ensure backward compatibility, we'll:
1. Keep old module locations but have them import from new locations
2. Maintain the same class and function signatures
3. Add deprecation warnings to old locations indicating the recommended import path

## Testing Strategy

Each phase will be tested to ensure:
1. All existing functionality works as before
2. All scripts can run without modifications
3. The test suite passes with the same results