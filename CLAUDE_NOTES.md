# Claude Notes: Neural Plasticity Experiment

## Key Points to Remember

1. **ALWAYS ACTIVATE VIRTUAL ENVIRONMENT**:
   ```bash
   source .venv/bin/activate
   ```

2. **What This Experiment Is REALLY About**:
   - Dynamic adaptation of transformer architecture through neural plasticity
   - NOT using magic numbers or hardcoded schedules
   - Making mathematical decisions about when to prune based on entropy and gradients
   - Using real models from HuggingFace and real datasets
   - Demonstrating true dynamic decision-making, not simulated responses

3. **Core Scripts**: 
   - `scripts/run_neural_plasticity.py` - Main experiment runner
   - `utils/neural_plasticity/dashboard/dashboard_demo.py` - Dashboard demonstration
   - Both support real-time monitoring and visualization

4. **Complete Experiment Flow**:
   1. Warmup training until loss stabilizes (detected mathematically)
   2. Analysis of attention patterns and head importance
   3. Pruning based on head importance metrics (entropy + gradients)
   4. Inference evaluation with baseline vs. pruned model comparison
   5. Comprehensive visualization via HTML dashboard & wandb

5. **Next Steps**:
   - Implement multi-cycle pruning with stabilization between cycles
   - Add fine-tuning after pruning to recover performance
   - Create advanced dashboard for tracking performance across cycles
   - Implement model growth capabilities alongside pruning
   - Port to Colab for longer runs on T4 GPU

6. **Important Implementation Details**:
   - Real-time dashboard works both locally and via wandb
   - Multiple visualization types: heatmaps, loss curves, architecture diagrams
   - Baseline model preservation for direct comparison of pruned vs. unpruned
   - Comprehensive inference evaluation phase
   - Support for sharing dashboard for collaborative analysis

7. **Files Overview**:
   - `utils/neural_plasticity/experiment.py`: Core experiment runner
   - `utils/neural_plasticity/dashboard/wandb_integration.py`: Dashboard integration
   - `utils/neural_plasticity/dashboard/dashboard_demo.py`: Dashboard demo
   - `scripts/run_neural_plasticity.py`: Main experiment script
   - `utils/neural_plasticity/dashboard/colab_integration.py`: Colab support

8. **Commands to Run Experiment**:
   ```bash
   # Activate virtual environment
   source .venv/bin/activate
   
   # Run dashboard demo (offline mode)
   python -m utils.neural_plasticity.dashboard.dashboard_demo
   
   # Run dashboard demo (online mode)
   WANDB_MODE=online python -m utils.neural_plasticity.dashboard.dashboard_demo --online
   
   # Run full experiment
   python scripts/run_neural_plasticity.py --model_name distilgpt2 --pruning_strategy entropy --pruning_level 0.2 --use_dashboard
   
   # View standalone HTML dashboard
   open output/neural_plasticity_<timestamp>/dashboard.html
   ```

## What We've Implemented (April 2025)

1. **Comprehensive Real-time Dashboard**:
   - Complete integration with Weights & Biases for experiment tracking
   - Standalone HTML dashboard for offline viewing
   - Support for collaboration through shareable links
   - Real-time console output for monitoring progress

2. **Enhanced Experiment Flow**:
   - Proper phase transitions (setup → warmup → analysis → pruning → evaluation)
   - Baseline model preservation for accurate comparison
   - Visualization of all experiment phases
   - Detailed inference evaluation of pruned vs. baseline models

3. **Advanced Visualization Capabilities**:
   - Interactive attention heatmaps
   - Pruning decision visualization (before/after)
   - Loss curves and performance metrics
   - Side-by-side text generation comparison
   - Perplexity evaluation charts

4. **Collaboration Features**:
   - Shareable dashboard links for team analysis
   - Automatic browser opening for local viewing
   - Support for both online and offline modes
   - Exportable HTML reports

## Dashboard Analysis Guide

When analyzing the dashboard results, focus on:

1. **Warmup Phase**: 
   - How quickly does the loss stabilize?
   - Is there a clear plateau in the loss curve?

2. **Entropy Heatmap**:
   - Which heads show the highest entropy values?
   - Is there a pattern to high-entropy heads across layers?

3. **Pruning Decisions**:
   - Which layers had the most heads pruned?
   - Does pruning target specific patterns or is it evenly distributed?

4. **Performance Impact**:
   - How much did perplexity increase after pruning?
   - Is the model size reduction worth the performance trade-off?

5. **Text Generation**:
   - How does the style/content differ between baseline and pruned models?
   - Is the generated text still coherent and meaningful?

The dashboard provides a holistic view of the neural plasticity process, from initial training through pruning to final evaluation.