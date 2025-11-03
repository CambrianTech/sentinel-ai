# Neural Plasticity Consolidation Plan

## Current State Analysis

Based on the repository analysis, we have three separate implementations of Neural Plasticity:

1. **In `/sentinel/` package (primary implementation)**:
   - `/sentinel/plasticity/plasticity_loop.py`: Core implementation with well-structured classes
   - `/sentinel/pruning/plasticity_controller.py`: Controller for neural plasticity
   - Other supporting modules in the appropriate structure

2. **In `/utils/neural_plasticity/` (duplicated functionality)**:
   - Core neural plasticity algorithms
   - Visualization utilities
   - Experiment runner

3. **Scattered scripts in `/scripts/`**:
   - 30+ scripts with redundant functionality
   - Many temporary fix scripts
   - Multiple "run" scripts that do similar things

## Consolidation Strategy

The primary goal is to consolidate all neural plasticity functionality into the proper `sentinel` package structure and remove redundant implementations.

### 1. Primary Implementation

The main implementation should be in the `sentinel` package, following the existing structure:

```
/sentinel/
  /plasticity/
    __init__.py
    plasticity_loop.py          # Core plasticity loop logic
    controller/
      __init__.py
      rl_controller.py         # Reinforcement learning controller
    defrag_heads.py            # Head defragmentation utilities
    entropy_journal.py         # Entropy tracking
    function_tracking.py       # Function usage tracking
    sleep_cycle.py             # Sleep phase management
    stress_protocols/          # Testing protocols
      __init__.py
      task_alternation.py
      task_suite.py
  /pruning/
    __init__.py
    plasticity_controller.py   # Controller for dynamic pruning
    dual_mode_pruning.py       # Adaptive/compressed pruning modes
    entropy_magnitude.py       # Entropy and magnitude based pruning
```

### 2. Runner Scripts

There should be a minimal set of entrypoint scripts:

```
/scripts/
  /neural_plasticity/
    __init__.py
    run_experiment.py          # Main experiment runner
    run_simple_test.py         # Simple test runner
    visualize_results.py       # Results visualization
```

### 3. Dashboard/Visualization

Dashboard and visualization utilities should be properly organized:

```
/sentinel/
  /visualization/
    __init__.py
    dashboard_generator.py     # Dashboard HTML generation
    entropy_rhythm_plot.py     # Entropy visualization
```

## Action Plan

1. **Keep Primary Implementation**:
   - `/sentinel/plasticity/plasticity_loop.py`
   - `/sentinel/pruning/plasticity_controller.py`
   - All supporting modules in the sentinel package

2. **Create Main Entry Points**:
   - Fix `/scripts/neural_plasticity/full_experiment.py` to use the sentinel implementation
   - Fix `/scripts/neural_plasticity/simple_experiment.py` to use the sentinel implementation

3. **Create Documentation**:
   - Add a comprehensive README in `/sentinel/plasticity/`
   - Create usage examples in `/scripts/neural_plasticity/examples/`

4. **Clean Up**:
   - Remove all redundant scripts from `/scripts/` root
   - Consolidate visualization code to proper location
   - Remove duplicated implementations in `/utils/neural_plasticity/`

5. **Updates for Colab Support**:
   - Ensure the implementation works in both local and Colab environments
   - Add environment detection to automatically configure for Colab

## Implementation Order

1. Fix and test the main entry points
2. Create documentation and examples
3. Clean up redundant files
4. Verify functionality in Colab