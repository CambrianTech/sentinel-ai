#!/usr/bin/env python
# Fix split header issues with last letter on new line

import json
import re
from pathlib import Path

def fix_split_headers(notebook_path):
    """Fix header issues where last letter appears on a new line."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Maps for fixed headers
        header_fixes = {
            "# Configure the Experimen": "# Configure the Experiment",
            "# Load Model and Datase": "# Load Model and Dataset",
            "# Define Evaluation Functio": "# Define Evaluation Function",
            "## Run Model Warm-u": "## Run Model Warm-up",
            "## Create Neural Plasticity Controlle": "## Create Neural Plasticity Controller",
            "## Collect Initial Head Metric": "## Collect Initial Head Metrics",
            "## Training with Neural Plasticit": "## Training with Neural Plasticity",
            "## Visualize Training Progres": "## Visualize Training Progress",
            "## Generate Text with Final Mode": "## Generate Text with Final Model",
            "## Try Different Prompt": "## Try Different Prompts",
            "## What is Neural Plasticit": "## What is Neural Plasticity"
        }
        
        # Track fixes applied
        fixes_applied = 0
        headers_examined = 0
        
        # Process each markdown cell
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'markdown':
                # Check source structure
                source_lines = cell['source']
                fixed_sources = []
                
                # Print debugging for certain cells
                if 2 <= i <= 10:
                    print(f"\nCell {i} source ({len(source_lines)} lines):")
                    for j, line in enumerate(source_lines[:5]):
                        print(f"  {j}: {repr(line)}")
                
                # Logic to rebuild source properly
                contents = ''.join(source_lines)
                
                # Check if this contains any problematic headers
                found_fix = False
                for bad_header, good_header in header_fixes.items():
                    if bad_header in contents:
                        # Get the next character that should be part of the header
                        pattern = re.escape(bad_header) + r'\s*\n\s*([a-z])'
                        match = re.search(pattern, contents)
                        if match:
                            last_char = match.group(1)
                            # We found a split header!
                            print(f"Found split header: {bad_header} + '{last_char}'")
                            
                            # Replace the split header with proper header and content
                            fixed_content = contents.replace(bad_header + '\n\n' + last_char, good_header)
                            
                            # For proper formatting, make sure header is followed by blank line
                            if not re.search(re.escape(good_header) + r'\n\n', fixed_content):
                                fixed_content = fixed_content.replace(good_header, good_header + '\n\n')
                            
                            # Update content
                            cell['source'] = [fixed_content]
                            found_fix = True
                            fixes_applied += 1
                            print(f"Fixed split header in cell {i}")
                            break
                
                # Last resort - look for any split headers with pattern: "# Title\n\nt"
                if not found_fix:
                    headers_examined += 1
                    header_pattern = r'(#+\s+[A-Za-z ]+)[a-zA-Z]\s*\n\s*([a-z])'
                    if re.search(header_pattern, contents):
                        match = re.search(header_pattern, contents)
                        if match:
                            print(f"Found generic split header in cell {i}: {match.group(0)}")
                            # Fix this header
                            fixed_content = re.sub(
                                header_pattern,
                                r'\1\2',  # Join the header parts
                                contents
                            )
                            # Update content
                            cell['source'] = [fixed_content]
                            fixes_applied += 1
                            print(f"Fixed generic split header in cell {i}")
        
        # Provide a final hardcoded fix to ensure headers are correct
        hardcoded_cells = {
            2: "# Configure the Experiment\n\nLet's set up our configuration for the neural plasticity experiment",
            5: "# Load Model and Dataset\n\nNow we'll load the model and prepare the dataset for training",
            7: "# Define Evaluation Function\n\nLet's define a function to evaluate our model's performance",
            9: "## Run Model Warm-up\n\nBefore measuring baseline performance and applying neural plasticity, we'll run a brief warm-up phase to get initial attention patterns and stabilize metrics.",
            13: "## Create Neural Plasticity Controller\n\nNow we'll create our neural plasticity controller that will monitor attention heads and make pruning decisions.",
            18: "## Collect Initial Head Metrics\n\nLet's look at the initial head metrics to establish our baseline.",
            20: "## Training with Neural Plasticity\n\nNow let's train the model with neural plasticity enabled, allowing it to adaptively prune and restore attention heads.",
            31: "## Visualize Training Progress\n\nLet's visualize the training history to see how neural plasticity affected the model.",
            34: "## Generate Text with Final Model\n\nLet's generate text with our plasticity-enhanced model to see the results.",
            38: "## Try Different Prompts\n\nLet's try generating text with different prompts to see how the model performs."
        }
        
        # Apply the hardcoded fixes
        for cell_idx, content in hardcoded_cells.items():
            if cell_idx < len(notebook['cells']):
                cell = notebook['cells'][cell_idx]
                if cell['cell_type'] == 'markdown':
                    cell['source'] = [content]
                    fixes_applied += 1
                    print(f"Applied hardcoded fix to cell {cell_idx}")
        
        # Save the notebook
        if fixes_applied > 0:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1)
            print(f"Applied {fixes_applied} fixes to {headers_examined} examined headers")
            return True
        else:
            print("No split headers found")
            return False
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_split_headers(notebook_path)