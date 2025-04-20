#!/usr/bin/env python
"""
Validate the NeuralPlasticityDemo.ipynb notebook after fixes

This script validates the notebook structure and checks for common issues
related to tensor handling, visualization, and Colab compatibility.
"""

import nbformat
import sys
import re
import os

def validate_notebook(notebook_path):
    """
    Validate a Jupyter notebook for common issues.
    
    Args:
        notebook_path: Path to the notebook
    
    Returns:
        Dictionary with validation results
    """
    print(f"Validating notebook: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    
    issues = {
        "detach_cpu_errors": [],
        "clim_errors": [],
        "undefined_monitor": [],
        "matplotlib_issues": [],
        "duplicate_imports": [],
        "tensor_viz_issues": [],
        "update_issues": [],
        "missing_imports": [],
        "other_issues": []
    }
    
    # Check for proper imports
    has_matplotlib_inline = False
    has_visualization_imports = False
    monitor_created = False
    
    # First pass: Check key elements
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            # Check for matplotlib inline
            if "%matplotlib inline" in cell.source:
                has_matplotlib_inline = True
            
            # Check for visualization imports
            if "from utils.colab.visualizations import" in cell.source:
                has_visualization_imports = True
            
            # Check for monitor creation
            if "pruning_monitor = TrainingMonitor(" in cell.source:
                monitor_created = True
    
    # If these key elements are missing, add them to issues
    if not has_matplotlib_inline:
        issues["matplotlib_issues"].append("Missing %matplotlib inline directive")
    
    if not has_visualization_imports:
        issues["missing_imports"].append("Missing import from utils.colab.visualizations")
    
    if not monitor_created:
        issues["undefined_monitor"].append("Training monitor is not defined")
    
    # Second pass: Check each cell for specific issues
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            # Check for detach/cpu issues
            if re.search(r"\.detach\(\.detach\(\)", cell.source):
                issues["detach_cpu_errors"].append(i)
                
            if re.search(r"\.cpu\(\)\.numpy\(\)\.cpu\(\)\.numpy\(\)", cell.source):
                issues["detach_cpu_errors"].append(i)
            
            # Check for tensor visualization errors
            # 1. Check for plt.imshow(...) calls without detach().cpu().numpy()
            for tensor_var in ["all_entropies", "grad_norms", "pruning_mask", "entropy_values", "grad_data", "entropy_data", "attn"]:
                if re.search(fr"plt\.imshow\(\s*{tensor_var}\s*[,)]", cell.source):
                    issues["tensor_viz_issues"].append(f"Cell {i}: plt.imshow({tensor_var}) without detach().cpu().numpy()")
                
                if re.search(fr"ax\.imshow\(\s*{tensor_var}\s*[,)]", cell.source):
                    issues["tensor_viz_issues"].append(f"Cell {i}: ax.imshow({tensor_var}) without detach().cpu().numpy()")
            
            # Check for plt.clim issues
            if re.search(r"plt\.clim\(0, 1\.0\).*?\.numpy\(\)", cell.source):
                issues["clim_errors"].append(i)
                
            if re.search(r"plt\.clim\(0, 1\.0\).*?# Ensure proper.*?# Ensure proper", cell.source):
                issues["clim_errors"].append(i)
            
            # Check for monitor usage without definition
            if "pruning_monitor.update" in cell.source and not monitor_created:
                issues["undefined_monitor"].append(i)
            
            # Check for duplicate imports
            if "# Import visualization utilities from utils.colab" in cell.source:
                if cell.source.count("# Import visualization utilities from utils.colab") > 1:
                    issues["duplicate_imports"].append(f"Cell {i}: Multiple visualization utility imports")
            
            # Check for commented out monitor updates
            if "# pruning_monitor.update_metrics" in cell.source:
                issues["update_issues"].append(f"Cell {i}: Commented out monitor update")
            
            # Check for broken update calls
            if "pruning_monitor.update(" in cell.source:
                issues["update_issues"].append(f"Cell {i}: Using deprecated update() method instead of update_metrics()")
            
            # Check for (removed) text
            if "(removed)" in cell.source:
                issues["other_issues"].append(f"Cell {i} contains '(removed)' text")
    
    # Third pass: Check for cell execution order and dependencies
    defined_vars = set()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            # Extract variable definitions
            var_defs = re.findall(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=", cell.source, re.MULTILINE)
            for var in var_defs:
                defined_vars.add(var)
            
            # Check for variables used before definition
            var_uses = re.findall(r"[^a-zA-Z0-9_]([a-zA-Z_][a-zA-Z0-9_]*)\s*[(.[]", cell.source)
            for var in var_uses:
                # Skip common builtins and imports
                if var not in defined_vars and var not in ["plt", "np", "torch", "print", "len", "range", "enumerate", "int", "float", "str", "list", "dict", "set", "tuple"]:
                    if not any(i.startswith("import " + var) for i in defined_vars) and not any("import " + var in c.source for c in nb.cells if c.cell_type == "code" and nb.cells.index(c) < i):
                        issues["other_issues"].append(f"Cell {i}: Potential use of undefined variable '{var}'")
    
    # Fourth pass: Check for device handling
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            # Check for proper device handling with CUDA tensors
            if "device = torch.device" in cell.source and ".to(device)" in cell.source:
                # Good: Has device definition and uses it
                pass
            elif ".cuda()" in cell.source and ".cpu()" not in cell.source:
                issues["other_issues"].append(f"Cell {i}: Uses .cuda() without .cpu() for visualization")
    
    # Print validation report
    print("\nValidation Results:")
    all_clean = True
    
    for issue_type, instances in issues.items():
        if isinstance(instances, list) and instances:
            all_clean = False
            if issue_type == "detach_cpu_errors" or issue_type == "clim_errors" or issue_type == "undefined_monitor":
                print(f"- {issue_type}: Found in cells {', '.join(map(str, instances))}")
            else:
                print(f"- {issue_type}: {len(instances)} issues")
                for issue in instances:
                    print(f"  * {issue}")
    
    if all_clean:
        print("✅ No issues found! Notebook looks good.")
        
        # Check if the notebook can be safely run in a Colab environment
        if has_matplotlib_inline and has_visualization_imports and monitor_created:
            print("\n🚀 Notebook appears to be Colab-ready!")
            print("- Has %matplotlib inline")
            print("- Imports visualization utilities")
            print("- Creates monitoring widgets")
            print("- No tensor visualization issues detected")
        else:
            print("\n⚠️ Notebook may not be fully Colab-compatible.")
    else:
        print("❌ Issues found in the notebook. Please fix them using fix_neural_plasticity_demo.py")
        
    return issues

def check_implementation_files():
    """Check that the required implementation files exist and have proper imports"""
    required_files = [
        "utils/colab/visualizations.py",
        "utils/colab/__init__.py"
    ]
    
    for path in required_files:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)
        if not os.path.exists(full_path):
            print(f"⚠️ Required file not found: {path}")
            return False
    
    return True

if __name__ == "__main__":
    # Get notebook path from command line or use default
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else "colab_notebooks/NeuralPlasticityDemo.ipynb"
    
    # Check implementation files
    print("Checking required implementation files...")
    implementation_ok = check_implementation_files()
    if not implementation_ok:
        print("Warning: Some required implementation files are missing.")
    
    # Validate the notebook
    issues = validate_notebook(notebook_path)
    
    # Return non-zero exit code if there are issues
    if any(len(instances) > 0 for instances in issues.values()):
        sys.exit(1)
    sys.exit(0)