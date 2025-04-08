#!/usr/bin/env python
# Fix all markdown headers in notebook

import json
import re
from pathlib import Path

def fix_all_headers(notebook_path):
    """Fix all markdown headers in notebook by manually checking each cell."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Define a list of known problematic headers to fix
        header_fixes = [
            ("# Configure the Experiment", "# Configure the Experiment\n\nLet's set up our configuration for the neural plasticity experiment"),
            ("# Load Model and Dataset", "# Load Model and Dataset\n\nNow we'll load the model and prepare the dataset for training"),
            ("# Define Evaluation Function", "# Define Evaluation Function\n\nLet's define a function to evaluate our model's performance"),
            ("## Run Model Warm-up", "## Run Model Warm-up\n\nBefore measuring baseline performance and applying neural plasticity, we'll run a brief warm-up phase to get initial attention patterns and stabilize metrics."),
            ("## Create Neural Plasticity Controller", "## Create Neural Plasticity Controller\n\nNow we'll create our neural plasticity controller that will monitor attention heads and make pruning decisions."),
            ("## Collect Initial Head Metrics", "## Collect Initial Head Metrics\n\nLet's look at the initial head metrics to establish our baseline."),
            ("## Training with Neural Plasticity", "## Training with Neural Plasticity\n\nNow let's train the model with neural plasticity enabled, allowing it to adaptively prune and restore attention heads."),
            ("## Visualize Training Progress", "## Visualize Training Progress\n\nLet's visualize the training history to see how neural plasticity affected the model."),
            ("## Generate Text with Final Model", "## Generate Text with Final Model\n\nLet's generate text with our plasticity-enhanced model to see the results."),
            ("## Try Different Prompts", "## Try Different Prompts\n\nLet's try generating text with different prompts to see how the model performs."),
        ]
        
        # Fix count
        fixes_applied = 0
        
        # Process each markdown cell
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'markdown':
                content = ''.join(cell['source'])
                
                # Check for header issues
                header_match = re.search(r'^#+\s+[^#\n]+', content)
                if header_match:
                    header = header_match.group(0)
                    
                    # Check if this header needs fixing
                    for bad_header, good_header in header_fixes:
                        if bad_header in header:
                            print(f"Found problematic header in cell {i}: {repr(header)}")
                            
                            # Replace with fixed header
                            new_content = content.replace(header, good_header)
                            
                            # Fix split headers like "## What is Neural Plasticity\n\n?\n\n"
                            new_content = re.sub(r'(##\s+[^#\n?!.]+)\s*\n\s*([?!.])', r'\1\2\n\n', new_content)
                            
                            # Update cell
                            cell['source'] = [new_content]
                            fixes_applied += 1
                            print(f"Fixed header in cell {i}")
                            break
        
        # Save the notebook if changes were made
        if fixes_applied > 0:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1)
            print(f"Applied {fixes_applied} header fixes")
            return True
        else:
            print("No header issues found")
            return False
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_all_headers(notebook_path)