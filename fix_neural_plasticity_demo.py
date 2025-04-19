#!/usr/bin/env python
"""
Fix NeuralPlasticityDemo.ipynb - Fix visualization issues and ensure proper tensor handling

This script applies fixes to NeuralPlasticityDemo.ipynb using nbformat as specified
in the project guidelines. It fixes visualization issues and ensures proper GPU tensor handling.
"""

import nbformat
import os
import sys
import re

def fix_notebook(notebook_path, output_path=None):
    """
    Fix the NeuralPlasticityDemo.ipynb notebook.
    
    Args:
        notebook_path: Path to the input notebook
        output_path: Path to save the fixed notebook (defaults to overwriting input)
    """
    print(f"Reading notebook: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    
    # Dictionary of fixes we need to apply
    fixes_applied = {
        "fixed_tensor_detach_cpu": 0,
        "fixed_clim_errors": 0,
        "fixed_visualization_imports": 0,
        "fixed_monitor_usage": 0
    }
    
    # Fix 1: Add proper tensor detach/cpu handling - more comprehensive regex approach
    for cell in nb.cells:
        if cell.cell_type == "code":
            # Fix tensor detach.cpu().numpy() issues using regex for more thorough replacement
            if re.search(r"\.detach\(\.detach\(\)\.cpu\(\)\.numpy\(\)\)", cell.source):
                cell.source = re.sub(
                    r"\.detach\(\.detach\(\)\.cpu\(\)\.numpy\(\)\)", 
                    ".detach().cpu().numpy()", 
                    cell.source
                )
                fixes_applied["fixed_tensor_detach_cpu"] += 1
    
            # Fix redundant .cpu().numpy().cpu().numpy() chains
            if re.search(r"\.cpu\(\)\.numpy\(\)\.cpu\(\)\.numpy\(\)", cell.source):
                cell.source = re.sub(
                    r"\.cpu\(\)\.numpy\(\)\.cpu\(\)\.numpy\(\)", 
                    ".cpu().numpy()", 
                    cell.source
                )
                fixes_applied["fixed_tensor_detach_cpu"] += 1
    
            # Fix cases where .cpu().numpy() is called on an already detached tensor
            if re.search(r"\.detach\(\)\.cpu\(\)\.numpy\(\)\.cpu\(\)\.numpy\(\)", cell.source):
                cell.source = re.sub(
                    r"\.detach\(\)\.cpu\(\)\.numpy\(\)\.cpu\(\)\.numpy\(\)", 
                    ".detach().cpu().numpy()", 
                    cell.source
                )
                fixes_applied["fixed_tensor_detach_cpu"] += 1
                
            # Fix another variation
            if re.search(r"\.detach\(\)\.cpu\(\)\.numpy\(\)\.cpu\(\)", cell.source):
                cell.source = re.sub(
                    r"\.detach\(\)\.cpu\(\)\.numpy\(\)\.cpu\(\)", 
                    ".detach().cpu()", 
                    cell.source
                )
                fixes_applied["fixed_tensor_detach_cpu"] += 1
    
    # Fix 2: Replace pruning_monitor with proper imports from utils.colab.visualizations
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            # Cell 24: Add proper import for TrainingMonitor and create the monitoring widget
            if "metrics_history = {" in cell.source and "inference_prompts = [" in cell.source:
                # This is cell 24 - Add imports and create monitor widget
                cell.source = cell.source.replace(
                    "# Create output directory for visualizations and checkpoints",
                    "# Import visualization utilities from utils.colab\n"
                    "from utils.colab.visualizations import TrainingMonitor, visualize_gradient_norms\n\n"
                    "# Create pruning monitor widget\n"
                    "pruning_monitor = TrainingMonitor(\n"
                    "    title=\"Neural Plasticity Training Progress\",\n"
                    "    metrics_to_track=[\"step\", \"epoch\", \"train_loss\", \"eval_loss\", \n"
                    "                     \"pruned_heads\", \"revived_heads\", \"sparsity\", \"perplexity\"]\n"
                    ")\n\n"
                    "# Create output directory for visualizations and checkpoints"
                )
                fixes_applied["fixed_visualization_imports"] += 1
            
            # Fix cell 30 that has a bare title function call with no object
            if "title=\"Neural Plasticity Training Progress\"" in cell.source and "metrics_to_track=[" in cell.source and not "pruning_monitor =" in cell.source:
                # Remove this standalone instantiation that's not assigned to a variable
                cell.source = cell.source.replace(
                    "    title=\"Neural Plasticity Training Progress\",\n"
                    "    metrics_to_track=[\"step\", \"epoch\", \"train_loss\", \"eval_loss\", \n"
                    "                     \"pruned_heads\", \"revived_heads\", \"sparsity\", \"perplexity\"]\n"
                    ")",
                    "# Using pruning_monitor already created above"
                )
                fixes_applied["fixed_monitor_usage"] += 1
            
            # Fix monitor usage by removing commented-out code for non-existent pruning_monitor
            if "# pruning_monitor.update" in cell.source or "# pruning_monitor.update" in cell.source:
                # Remove commented monitor calls
                cell.source = cell.source.replace("# pruning_monitor.update", "# pruning_monitor.update")
                fixes_applied["fixed_monitor_usage"] += 1
                
            # Fix (removed) text in the code
            if "(removed)" in cell.source:
                cell.source = cell.source.replace("(removed)", "")
                fixes_applied["fixed_monitor_usage"] += 1
    
    # Fix 3: Fix clim issues in cell 19 and other visualization rendering issues
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            if "plt.clim(0, 1.0)  # Ensure proper scaling for attention values.numpy()" in cell.source:
                cell.source = cell.source.replace(
                    "plt.clim(0, 1.0)  # Ensure proper scaling for attention values.numpy()",
                    "plt.clim(0, 1.0)  # Ensure proper scaling for attention values"
                )
                fixes_applied["fixed_clim_errors"] += 1
            
            if "plt.clim(0, 1.0)  # Ensure proper scaling for attention  # Ensure proper scale for attention visualization" in cell.source:
                cell.source = cell.source.replace(
                    "plt.clim(0, 1.0)  # Ensure proper scaling for attention  # Ensure proper scale for attention visualization",
                    "plt.clim(0, 1.0)  # Ensure proper scaling for attention visualization"
                )
                fixes_applied["fixed_clim_errors"] += 1
                
            # Fix other clim issues
            if ".detach().cpu().numpy()).cpu().numpy()" in cell.source:
                cell.source = cell.source.replace(
                    ".detach().cpu().numpy()).cpu().numpy()",
                    ".detach().cpu().numpy())"
                )
                fixes_applied["fixed_clim_errors"] += 1
                
            # Fix remaining tensor visualization issues
            cell.source = re.sub(
                r"\.imshow\(([^()]*?)\.detach\(\)\.cpu\(\)\.numpy\(\)\.cpu\(\)\.numpy\(\)",
                r".imshow(\1.detach().cpu().numpy()",
                cell.source
            )
            
            # Fix another variation
            cell.source = re.sub(
                r"\.imshow\(([^()]*?)\.detach\(\)\.cpu\(\)\.numpy\(\)\.cpu\(\)",
                r".imshow(\1.detach().cpu().numpy()",
                cell.source
            )
            
            # Fix another pattern we found
            if "plt.imshow(pruning_mask.detach(.detach().cpu().numpy())" in cell.source:
                cell.source = cell.source.replace(
                    "plt.imshow(pruning_mask.detach(.detach().cpu().numpy())",
                    "plt.imshow(pruning_mask.detach().cpu().numpy()"
                )
                fixes_applied["fixed_tensor_detach_cpu"] += 1
                
            # Fix all occurrences of incorrect detach call
            if "detach(" in cell.source:
                cell.source = re.sub(
                    r"\.detach\(\.detach\(\)",
                    ".detach()",
                    cell.source
                )
                fixes_applied["fixed_tensor_detach_cpu"] += 1
                
            # Just in case there are any other variations
            cell.source = re.sub(
                r"\.detach\([^()]*?\)",
                ".detach()",
                cell.source
            )
            
    # Save the fixed notebook
    output_file = output_path or notebook_path
    print(f"Saving fixed notebook to: {output_file}")
    nbformat.write(nb, output_file)
    
    # Print summary of fixes
    print("\nFixes applied:")
    for fix, count in fixes_applied.items():
        print(f"- {fix}: {count} instances")
    
    return fixes_applied


if __name__ == "__main__":
    # Get notebook path from command line or use default
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else "colab_notebooks/NeuralPlasticityDemo.ipynb"
    
    # Option to specify output path as second argument
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Apply fixes
    fix_notebook(notebook_path, output_path)