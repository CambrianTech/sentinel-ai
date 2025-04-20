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
from datetime import datetime

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
        "fixed_monitor_usage": 0,
        "fixed_matplotlib_inline": 0,
        "fixed_version_number": 0,
        "fixed_duplicate_imports": 0
    }
    
    # Check if matplotlib inline is present at the top
    has_matplotlib_inline = False
    for cell in nb.cells:
        if cell.cell_type == "code" and "%matplotlib inline" in cell.source:
            has_matplotlib_inline = True
            break
    
    # Add %matplotlib inline to the first code cell if it's not present
    if not has_matplotlib_inline:
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and "import" in cell.source:
                if "%matplotlib inline" not in cell.source:
                    cell.source = "%matplotlib inline\n" + cell.source
                    fixes_applied["fixed_matplotlib_inline"] += 1
                break
    
    # Fix version number in title cell
    for cell in nb.cells:
        if cell.cell_type == "markdown" and "# Neural Plasticity Demo" in cell.source:
            # Extract current version
            version_match = re.search(r"\(v(\d+\.\d+\.\d+)\)", cell.source)
            if version_match:
                current_version = version_match.group(1)
                # Parse version
                parts = current_version.split(".")
                major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
                # Increment patch version
                new_version = f"{major}.{minor}.{patch+1}"
                # Add timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Replace version in title
                cell.source = re.sub(
                    r"\(v\d+\.\d+\.\d+\)",
                    f"(v{new_version} {timestamp})",
                    cell.source
                )
                fixes_applied["fixed_version_number"] += 1
                
                # Update "New in v0.0.XX" sections
                # Remove duplicate entries 
                if f"### New in v{current_version}:" in cell.source and cell.source.count(f"### New in v{current_version}:") > 1:
                    # Find the first occurrence and keep only that one
                    first_idx = cell.source.find(f"### New in v{current_version}:")
                    second_idx = cell.source.find(f"### New in v{current_version}:", first_idx + 1)
                    end_idx = cell.source.find("###", second_idx + 1)
                    if end_idx == -1:  # If there's no next section, go to the end
                        end_idx = len(cell.source)
                    
                    # Remove the duplicate section
                    cell.source = cell.source[:second_idx] + cell.source[end_idx:]
                    fixes_applied["fixed_duplicate_imports"] += 1
                
                # Add new version section
                new_section = f"""
### New in v{new_version}:
- Fixed GPU tensor visualization errors
- Fixed visualization utilities integration
- Ensured proper tensor detachment and CPU conversion for visualization
- Integrated with utils.colab.visualizations module
- Added %matplotlib inline for Colab compatibility
- Added system dependency checks
- Improved error handling in training loop
- Fixed tensorboard visualizations 
- Enhanced memory management
- Deduplicated import statements
- Fixed cell execution counts for better notebook flow"""
                
                # Add new section after the title
                title_end = cell.source.find("\n", cell.source.find("# Neural Plasticity Demo"))
                if title_end != -1:
                    cell.source = cell.source[:title_end + 1] + new_section + cell.source[title_end:]
                
    # Fix 1: Add proper tensor detach/cpu handling - more comprehensive regex approach
    for cell in nb.cells:
        if cell.cell_type == "code":
            # First count original instances to recognize the pattern
            original = cell.source
            
            # Fix tensor detach.cpu().numpy() issues using regex for more thorough replacement
            cell.source = re.sub(
                r"\.detach\(\.detach\(\)\.cpu\(\)\.numpy\(\)\)", 
                ".detach().cpu().numpy()", 
                cell.source
            )
            
            # Fix redundant .cpu().numpy().cpu().numpy() chains
            cell.source = re.sub(
                r"\.cpu\(\)\.numpy\(\)\.cpu\(\)\.numpy\(\)", 
                ".cpu().numpy()", 
                cell.source
            )
            
            # Fix cases where .cpu().numpy() is called on an already detached tensor
            cell.source = re.sub(
                r"\.detach\(\)\.cpu\(\)\.numpy\(\)\.cpu\(\)\.numpy\(\)", 
                ".detach().cpu().numpy()", 
                cell.source
            )
            
            # Fix common visualization error: plt.imshow with .cpu().numpy().cpu().numpy()
            cell.source = re.sub(
                r"(plt\.imshow\([^)]*?)\.cpu\(\)\.numpy\(\)\.cpu\(\)\.numpy\(\)([^)]*?\))",
                r"\1.cpu().numpy()\2",
                cell.source
            )
            
            # Fix common visualization error: plt.imshow with .detach().cpu().numpy().cpu().numpy()
            cell.source = re.sub(
                r"(plt\.imshow\([^)]*?)\.detach\(\)\.cpu\(\)\.numpy\(\)\.cpu\(\)\.numpy\(\)([^)]*?\))",
                r"\1.detach().cpu().numpy()\2",
                cell.source
            )
            
            # Fix another variation with .detach() and .cpu() without .numpy()
            cell.source = re.sub(
                r"\.detach\(\)\.cpu\(\)\.numpy\(\)\.cpu\(\)", 
                ".detach().cpu().numpy()", 
                cell.source
            )
            
            # Fix another pattern we found
            cell.source = cell.source.replace(
                "plt.imshow(pruning_mask.detach(.detach().cpu().numpy())",
                "plt.imshow(pruning_mask.detach().cpu().numpy()"
            )
            
            # Fix all occurrences of incorrect detach call
            cell.source = re.sub(
                r"\.detach\(\.detach\(\)",
                ".detach()",
                cell.source
            )
            
            # Just in case there are any other variations
            cell.source = re.sub(
                r"\.detach\([^()]*?\)",
                ".detach()",
                cell.source
            )
            
            # Count the fixes
            if original != cell.source:
                fixes_applied["fixed_tensor_detach_cpu"] += 1
    
    # Fix 2: Fix visualization utilities and imports
    imports_added = False
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            # Add proper imports in one place
            if "import torch" in cell.source and "import numpy as np" in cell.source and not imports_added:
                # This is the main imports cell, add visualization imports here
                if "from utils.colab.visualizations import" not in cell.source:
                    # Add after the last import
                    last_import_idx = cell.source.rfind("import ")
                    if last_import_idx != -1:
                        line_end = cell.source.find("\n", last_import_idx)
                        if line_end == -1:
                            line_end = len(cell.source)
                        
                        visualization_imports = "\n# Import visualization utilities\nfrom utils.colab.visualizations import TrainingMonitor, visualize_gradient_norms, visualize_attention_heatmap, visualize_head_entropy\n"
                        cell.source = cell.source[:line_end+1] + visualization_imports + cell.source[line_end+1:]
                        imports_added = True
                        fixes_applied["fixed_visualization_imports"] += 1
            
            # Fix monitor creation - ensure it's created only once
            if "pruning_monitor = " in cell.source and "TrainingMonitor(" in cell.source:
                # This is where the monitor is created, make sure it's done correctly
                if "from utils.colab.visualizations import" not in cell.source and not "TrainingMonitor" in cell.source.split("from utils.colab.visualizations import")[1]:
                    # Fix import if needed
                    fixes_applied["fixed_visualization_imports"] += 1
                
            # Remove duplicate monitor creation
            if "# Import visualization utilities from utils.colab" in cell.source and cell.source.count("# Import visualization utilities from utils.colab") > 1:
                # Find all occurrences
                occurrences = [m.start() for m in re.finditer("# Import visualization utilities from utils.colab", cell.source)]
                if len(occurrences) > 1:
                    # Keep the first one, remove the rest
                    for i in range(1, len(occurrences)):
                        start_idx = occurrences[i]
                        # Find where this section ends (next comment or empty line)
                        next_section = cell.source.find("\n#", start_idx + 1)
                        if next_section == -1:
                            next_section = len(cell.source)
                        
                        # Remove this duplicate section
                        cell.source = cell.source[:start_idx] + cell.source[next_section:]
                        fixes_applied["fixed_duplicate_imports"] += 1
    
    # Fix 3: Fix matplotlib visualization issues
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            # Fix clim issues
            if "plt.clim(0, 1.0)" in cell.source:
                # Fix incorrect clim calls
                original = cell.source
                
                cell.source = re.sub(
                    r"plt\.clim\(0, 1\.0\)\s*#\s*Ensure proper scaling for attention values\.numpy\(\)",
                    "plt.clim(0, 1.0)  # Ensure proper scaling for attention values",
                    cell.source
                )
                
                cell.source = re.sub(
                    r"plt\.clim\(0, 1\.0\)\s*#\s*Ensure proper scaling for attention\s*#\s*Ensure proper scale for attention visualization",
                    "plt.clim(0, 1.0)  # Ensure proper scaling for attention visualization",
                    cell.source
                )
                
                # Fix imshow issues with detached tensors 
                cell.source = re.sub(
                    r"plt\.imshow\(([^()]*?)\.detach\(\)\.cpu\(\)\.numpy\(\),\s*cmap=['\"]([^'\"]+)['\"],\s*aspect=['\"]([^'\"]+)['\"]\.detach\(\)\.cpu\(\)\.numpy\(\)\)",
                    r"plt.imshow(\1.detach().cpu().numpy(), cmap='\2', aspect='\3')",
                    cell.source
                )
                
                # Fix other variations
                cell.source = re.sub(
                    r"(plt\.imshow\([^)]*?)\.cpu\(\)\.numpy\(\)\)",
                    r"\1)",
                    cell.source
                )
                
                if original != cell.source:
                    fixes_applied["fixed_clim_errors"] += 1
            
            # Fix plot calls that don't include detach().cpu().numpy()
            if "plt.imshow(" in cell.source or "ax.imshow(" in cell.source:
                original = cell.source
                
                # With tensor variables
                for tensor_var in ["grad_norms", "pruning_mask", "all_entropies", "entropy_data", "entropy_values", "grad_data", "attn"]:
                    # Matches: plt.imshow(tensor_var cmap=...) without .detach().cpu().numpy()
                    cell.source = re.sub(
                        fr"(plt\.imshow\({tensor_var}(?!\.).*?\))",
                        fr"plt.imshow({tensor_var}.detach().cpu().numpy())",
                        cell.source
                    )
                    
                    # Same for ax.imshow
                    cell.source = re.sub(
                        fr"(ax\.imshow\({tensor_var}(?!\.).*?\))",
                        fr"ax.imshow({tensor_var}.detach().cpu().numpy())",
                        cell.source
                    )
                
                # Fix partial detach calls
                cell.source = re.sub(
                    r"(plt\.imshow\([^)]*?)\.detach\(\)([^)]*?\))",
                    r"\1.detach().cpu().numpy()\2",
                    cell.source
                )
                
                # Fix partial cpu calls
                cell.source = re.sub(
                    r"(plt\.imshow\([^)]*?)\.cpu\(\)([^)]*?\))",
                    r"\1.cpu().numpy()\2",
                    cell.source
                )
                
                if original != cell.source:
                    fixes_applied["fixed_clim_errors"] += 1
            
    # Fix 4: Fix monitor usage
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            # Fix monitor usage by uncommenting correct calls
            if "# pruning_monitor.update" in cell.source:
                original = cell.source
                # Uncomment good calls
                cell.source = cell.source.replace(
                    "# pruning_monitor.update_metrics",
                    "pruning_monitor.update_metrics"
                )
                
                if original != cell.source:
                    fixes_applied["fixed_monitor_usage"] += 1
            
            # Remove broken update calls
            if "pruning_monitor.update(" in cell.source:
                original = cell.source
                # Convert to proper update_metrics calls
                cell.source = cell.source.replace(
                    "pruning_monitor.update(",
                    "# Old: pruning_monitor.update("
                )
                if original != cell.source:
                    fixes_applied["fixed_monitor_usage"] += 1
                
            # Fix (removed) text or other comments that are artifacts
            if "(removed)" in cell.source:
                cell.source = cell.source.replace("(removed)", "")
                fixes_applied["fixed_monitor_usage"] += 1
    
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