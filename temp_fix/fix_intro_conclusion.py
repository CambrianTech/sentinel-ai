#!/usr/bin/env python
# Fix intro and conclusion sections

import json
from pathlib import Path

def fix_intro_conclusion(notebook_path):
    """Fix the intro and conclusion sections of the notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Fix intro section (cell 0)
    intro_content = """# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.21)

This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to dynamically prune and regrow attention heads during training based on utility metrics.

## What is Neural Plasticity?

Neural plasticity is the ability of neural networks to adapt their structure over time through pruning (removing unused connections) and regrowth (restoring useful connections). This mimics how biological brains form efficient neural pathways.

In this demo, we:
1. Track the entropy and gradient patterns of each attention head
2. Dynamically prune high-entropy, low-gradient heads (unfocused, less useful)
3. Selectively revive low-entropy, higher-gradient heads (potentially useful)
4. Visualize the "brain dynamics" over time

This allows models to form more efficient neural structures during training.

### New in v0.0.21:
- Fixed intro and conclusion section formatting
- Fixed cell character encoding issues
- Split large cells into focused, manageable sections

### New in v0.0.20:
- Fixed cell size issues by splitting large cells
- Fixed markdown formatting throughout the notebook
- Improved cell organization and readability
- Fixed entropy calculation to prevent zero values
- Added numerical stability improvements 
- Properly normalized attention patterns

### New in v0.0.17:
- Fixed visualization scaling to prevent extremely large plots
- Added data downsampling for large training runs
- Set explicit DPI control to maintain reasonable image sizes
- Improved epoch boundary visualization

### Previous in v0.0.16:
- Fixed critical pruning logic to correctly target heads with LOWEST gradient norms
- Added comprehensive attention pattern visualization with log scaling
- Fixed serialization error when saving checkpoints
- Added detailed gradient statistics for pruned vs. kept heads
- Enhanced gradient visualization to better highlight pruning decisions"""
    
    # Find the intro cell and update it
    if notebook['cells'][0]['cell_type'] == 'markdown':
        notebook['cells'][0]['source'] = intro_content.split('\n')
        print("Fixed intro section (cell 0)")
    
    # Fix conclusion section (last markdown cell)
    for i in range(len(notebook['cells'])-1, -1, -1):
        if notebook['cells'][i]['cell_type'] == 'markdown' and '## Conclusion' in ''.join(notebook['cells'][i]['source']):
            conclusion_content = """## Conclusion

In this notebook, we demonstrated Sentinel AI's neural plasticity system, which enables transformer models to dynamically prune and revive attention heads during training based on their utility.

Key findings:
1. The plasticity system successfully pruned high-entropy, low-gradient heads
2. Some heads were revived when they showed potential for useful learning
3. The final model achieved comparable quality with fewer active heads
4. The brain dynamics visualization shows how attention heads evolve over time

This approach mimics biological neural plasticity, where brains form efficient neural pathways by pruning unused connections and strengthening useful ones.

## Version History

- v0.0.21: Fixed intro and conclusion formatting, split large cells into focused sections, fixed cell character encoding issues
- v0.0.20: Split large cells into smaller sections, fixed markdown formatting throughout the notebook, improved code organization and readability
- v0.0.19: Fixed entropy calculation to prevent zero values, added numerical stability improvements, properly normalized attention patterns
- v0.0.17: Fixed visualization scaling to prevent extremely large plots, added data downsampling for training history, improved epoch visualization
- v0.0.16: Fixed critical pruning logic to target heads with lowest gradient norms, added comprehensive attention pattern visualization, fixed serialization issues
- v0.0.15: Improved warm-up phase to run until loss stabilizes with automatic detection, added comprehensive warm-up monitoring and stabilization metrics
- v0.0.14: Added compatibility fixes for visualization on different platforms, replaced Unicode markers with text-based markers, improved entropy visualization
- v0.0.13: Replaced DEMO_MAX_STEPS with MAX_STEPS_PER_EPOCH for better training control, allowing full epoch completion for improved model stability and visualization
- v0.0.12: Added integration with comprehensive adaptive plasticity visualization system, enhanced metrics dashboard, improved overlay markers for clarity
- v0.0.11: Added live visualization of gradient norms with pruning decisions during training, with side-by-side comparison of initial vs final state
- v0.0.10: Added unit tests for visualization functions to ensure reliability and correctness
- v0.0.9: Added new gradient visualization with pruning overlays (red X's, green plus, yellow warning)
- v0.0.8: Fixed entropy calculation issue by implementing gradient-only based pruning
- v0.0.7: Replaced fixed magic number thresholds with statistical approach using percentile-based pruning
- v0.0.6: Fixed bug in debug code (removed invalid 'verbose' parameter from collect_head_metrics call)
- v0.0.5: Significantly more aggressive pruning thresholds (HIGH_ENTROPY_THRESHOLD: 0.6→0.4, LOW_ENTROPY_THRESHOLD: 0.3→0.2, GRAD_THRESHOLD: 5e-5→1e-3)
- v0.0.4: Adjusted pruning thresholds for more aggressive pruning behavior (HIGH_ENTROPY_THRESHOLD: 0.8→0.6, LOW_ENTROPY_THRESHOLD: 0.4→0.3, GRAD_THRESHOLD: 1e-4→5e-5)
- v0.0.3: Removed hard-coded 200-step limit to allow full NUM_EPOCHS training
- v0.0.2: Added warmup phase to get more accurate baseline measurements, improved visualization of head metrics, fixed perplexity calculation issues
- v0.0.1: Initial implementation of neural plasticity demo"""
            
            notebook['cells'][i]['source'] = conclusion_content.split('\n')
            print(f"Fixed conclusion section (cell {i})")
            break
    
    # Save the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print("Saved notebook with fixed intro and conclusion sections")
    return True

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_intro_conclusion(notebook_path)