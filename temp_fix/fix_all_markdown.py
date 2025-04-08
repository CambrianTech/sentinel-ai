#!/usr/bin/env python
# Comprehensive fix for markdown formatting in NeuralPlasticityDemo.ipynb

import json
import re
from pathlib import Path

def fix_all_markdown(notebook_path):
    """Fix all markdown formatting issues in the notebook."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Process each markdown cell
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "markdown":
                # Get the full text content
                content = ''.join(cell["source"])
                
                # Fix 1: Ensure headers are properly formatted with newlines
                # Find all headers (##...) and ensure they have newlines after them
                content = re.sub(r'(#+[^\n]+)([^#\n])', r'\1\n\2', content)
                
                # Fix 2: Ensure list items have proper spacing
                content = re.sub(r'(\n\d+\.)([^\n])', r'\n\1 \2', content)
                
                # Fix 3: Replace ### with proper newlines
                content = content.replace('###', '\n###')
                
                # Fix 4: Ensure proper spacing around headers
                content = re.sub(r'([^\n])(#+\s+)', r'\1\n\n\2', content)
                content = re.sub(r'(#+[^\n]+)([^\n])', r'\1\n\n\2', content)
                
                # Fix 5: Special handling for cell 0 (introduction)
                if i == 0:
                    # Replace with properly formatted intro
                    content = """# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.23)

This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to dynamically prune and regrow attention heads during training based on utility metrics.

## What is Neural Plasticity?

Neural plasticity is the ability of neural networks to adapt their structure over time through pruning (removing unused connections) and regrowth (restoring useful connections). This mimics how biological brains form efficient neural pathways.

In this demo, we:
1. Track the entropy and gradient patterns of each attention head
2. Dynamically prune high-entropy, low-gradient heads (unfocused, less useful)
3. Selectively revive low-entropy, higher-gradient heads (potentially useful)
4. Visualize the "brain dynamics" over time

This allows models to form more efficient neural structures during training.

### New in v0.0.23:
- Fixed visualization issues causing excessively large images
- Reduced figure sizes and DPI settings
- Fixed cell splitting in controller section

### New in v0.0.22:
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
                
                # Special handling for other problematic cells
                elif "Run Model Warm-up" in content:
                    content = """## Run Model Warm-up

Before measuring baseline performance and applying neural plasticity, we'll run a brief warm-up phase to get initial attention patterns and stabilize metrics."""
                
                elif "Create Neural Plasticity Controller" in content:
                    content = """## Create Neural Plasticity Controller

Now we'll create our neural plasticity controller that will monitor attention heads and make pruning decisions."""
                
                elif "Collect Initial Head Metrics" in content:
                    content = """## Collect Initial Head Metrics

Let's look at the initial head metrics to establish our baseline."""
                
                elif "Training with Neural Plasticity" in content:
                    content = """## Training with Neural Plasticity

Now let's train the model with neural plasticity enabled, allowing it to adaptively prune and restore attention heads."""
                
                elif "Visualize Training Progress" in content:
                    content = """## Visualize Training Progress

Let's visualize the training history to see how neural plasticity affected the model."""
                
                elif "Generate Text with Final Model" in content:
                    content = """## Generate Text with Final Model

Let's generate text with our plasticity-enhanced model to see the results."""
                
                elif "Try Different Prompts" in content:
                    content = """## Try Different Prompts

Let's try generating text with different prompts to see how the model performs."""
                
                elif "Conclusion" in content:
                    # Fix conclusion cell - careful with version history
                    lines = content.split("\n")
                    fixed_lines = []
                    in_version_history = False
                    
                    for line in lines:
                        if "## Conclusion" in line:
                            fixed_lines.append("## Conclusion\n")
                            continue
                        elif "## Version History" in line:
                            fixed_lines.append("\n## Version History\n")
                            in_version_history = True
                            continue
                        
                        if in_version_history and line.strip().startswith("- v0.0."):
                            fixed_lines.append(line)
                        else:
                            fixed_lines.append(line)
                    
                    content = "\n".join(fixed_lines)
                    
                    # Make sure the conclusion has proper formatting
                    if "## Conclusion" in content and "demonstrated Sentinel AI" in content:
                        conclusion_text = """## Conclusion

In this notebook, we demonstrated Sentinel AI's neural plasticity system, which allows transformer models to dynamically adapt their structure during training. 

The key components we implemented were:
1. Tracking attention head entropy and gradient values
2. Dynamically pruning unfocused heads (high entropy, low gradient impact)  
3. Selectively reviving promising heads to balance performance
4. Visualizing the "brain dynamics" over time

This approach allows models to form more efficient neural pathways, similar to how biological brains optimize connections. The resulting model maintains performance while reducing computational requirements.

## Version History

- v0.0.23: Fixed visualization issues causing excessively large images, reduced figure sizes and DPI settings
- v0.0.22: Fixed intro and conclusion formatting, split large cells into focused sections, fixed cell character encoding issues
- v0.0.21: Fixed intro and conclusion formatting, split large cells into focused sections, fixed cell character encoding issues
- v0.0.20: Split large cells into smaller sections, fixed markdown formatting throughout the notebook, improved code organization and readability
- v0.0.19: Fixed entropy calculation to prevent zero values, added numerical stability improvements, properly normalized attention patterns
- v0.0.17: Fixed visualization scaling to prevent extremely large plots, added data downsampling for training history, improved epoch visualization
- v0.0.16: Fixed critical pruning logic to target heads with lowest gradient norms, added comprehensive attention pattern visualization, fixed serialization issues
- v0.0.15: Improved warm-up phase to run until loss stabilizes with automatic detection, added comprehensive warm-up monitoring and stabilization metrics
- v0.0.14: Added compatibility fixes for visualization on different platforms, replaced Unicode markers with text-based markers, improved entropy visualization
- v0.0.13: Fixed critical numerical stability issues, improved gradient tracking, and enhanced entropy calculation accuracy
- v0.0.12: Added comprehensive entropy visualization and improved tracking of pruned heads
- v0.0.11: Added support for head revival based on entropy and gradient criteria
- v0.0.10: Fixed loss scaling and batch normalization in adaptive transformer
- v0.0.9: Initial implementation of neural plasticity with entropy-based pruning"""
                        content = conclusion_text
                
                # Update the cell source - make sure each line is a separate entry in the list
                new_source = []
                for line in content.split('\n'):
                    new_source.append(line + '\n')
                
                # Remove the trailing newline from the last element if it exists
                if new_source and new_source[-1].endswith('\n'):
                    new_source[-1] = new_source[-1].rstrip('\n')
                
                cell["source"] = new_source
                print(f"Fixed markdown in cell {i}: {new_source[0][:30]}...")
        
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"Notebook saved with all markdown formatting fixed")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_all_markdown(notebook_path)