#!/usr/bin/env python
"""
Comprehensive validation for Colab notebooks

This script performs a thorough validation of Jupyter notebooks,
checking for common issues, best practices, and potential runtime problems.
"""

import nbformat
import sys
import re
import os
from typing import Dict, List, Set, Tuple, Any


def validate_cell_execution_order(notebook_path: str) -> Dict[str, Any]:
    """Validate the execution order of cells in a notebook"""
    print(f"Validating execution order: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    
    issues = {
        "execution_count_issues": [],
        "missing_execution_count": 0,
        "out_of_order_cells": []
    }
    
    prev_exec_count = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
            
        # Check if execution_count exists
        if not hasattr(cell, 'execution_count') or cell.execution_count is None:
            issues["missing_execution_count"] += 1
            continue
            
        # Check if execution is in order
        if cell.execution_count != 0 and cell.execution_count < prev_exec_count:
            issues["out_of_order_cells"].append((i, prev_exec_count, cell.execution_count))
            
        prev_exec_count = cell.execution_count if cell.execution_count != 0 else prev_exec_count
    
    return issues


def validate_import_statements(notebook_path: str) -> Dict[str, Any]:
    """Check for import issues in notebook cells"""
    print(f"Validating imports: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    
    issues = {
        "duplicate_imports": [],
        "import_in_multiple_cells": {},
        "missing_imports": set(),
        "unused_imports": set()
    }
    
    # Track all imports
    imports_by_cell = {}
    all_imports = set()
    imported_modules = set()
    
    # Pattern to find module usage (simplified)
    module_usage_pattern = r'([a-zA-Z][a-zA-Z0-9_]*)\.'
    
    # First pass: collect imports
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
            
        # Find import statements
        cell_imports = []
        for line in cell.source.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                cell_imports.append(line)
                
                # Extract module names
                if line.startswith('import '):
                    modules = line[7:].split(',')
                    for module in modules:
                        mod = module.strip().split(' as ')[0]
                        imported_modules.add(mod)
                        all_imports.add(mod)
                elif line.startswith('from '):
                    parts = line.split(' import ')
                    if len(parts) == 2:
                        mod = parts[0][5:].strip()
                        imported_modules.add(mod)
                        
                        # Handle aliased imports
                        for item in parts[1].split(','):
                            item = item.strip()
                            if ' as ' in item:
                                alias = item.split(' as ')[1].strip()
                                all_imports.add(alias)
                            else:
                                all_imports.add(item)
        
        if cell_imports:
            imports_by_cell[i] = cell_imports
    
    # Check for duplicate imports
    all_import_lines = []
    for cell_idx, imports in imports_by_cell.items():
        for imp in imports:
            if imp in all_import_lines:
                issues["duplicate_imports"].append((cell_idx, imp))
            else:
                all_import_lines.append(imp)
    
    # Check for modules imported in multiple cells
    module_to_cells = {}
    for cell_idx, imports in imports_by_cell.items():
        for imp in imports:
            if imp not in module_to_cells:
                module_to_cells[imp] = []
            module_to_cells[imp].append(cell_idx)
    
    for module, cells in module_to_cells.items():
        if len(cells) > 1:
            issues["import_in_multiple_cells"][module] = cells
    
    # Second pass: check for module usage
    used_modules = set()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
            
        # Find module usages
        matches = re.findall(module_usage_pattern, cell.source)
        used_modules.update(matches)
    
    # Check for unused imports
    for module in all_imports:
        if module not in used_modules and module != '*':
            issues["unused_imports"].add(module)
    
    # Check for missing imports (simplified - may have false positives)
    for module in used_modules:
        if module not in all_imports and not any(m.endswith(f'.{module}') for m in all_imports):
            # Skip common built-ins and local variables
            if module not in ['np', 'plt', 'torch', 'self', 'os', 'sys', 'math', 'random']:
                issues["missing_imports"].add(module)
    
    return issues


def validate_tensor_operations(notebook_path: str) -> Dict[str, Any]:
    """Check for tensor operation issues"""
    print(f"Validating tensor operations: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    
    issues = {
        "potential_cuda_issues": [],
        "tensor_manipulation_issues": [],
        "missing_detach_cpu": []
    }
    
    # Patterns to find potential issues
    cuda_patterns = [
        (r'\.cuda\(\)', "Explicit .cuda() call may fail in CPU-only environment"),
        (r'device\s*=\s*[\'"]cuda[\'"]', "Hardcoded cuda device may fail in CPU-only environment")
    ]
    
    visualization_patterns = [
        (r'plt\.(?:imshow|plot|bar|hist|scatter|pcolor|contour)\([^)]*?(?<!\.detach\(\)\.cpu\(\)\.numpy\(\))\)', 
         "Possible tensor visualization without detach().cpu().numpy()"),
        (r'\.detach\(\)\.numpy\(\)', "Using .detach().numpy() without .cpu() may fail on CUDA tensors")
    ]
    
    # Simple heuristic to check if a line involves visualization
    def is_visualization_context(line):
        return any(vis_func in line for vis_func in ['plt.imshow', 'plt.plot', 'plt.bar', 'plt.hist', 'plt.scatter'])
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
            
        # Check for CUDA issues
        for pattern, message in cuda_patterns:
            if re.search(pattern, cell.source):
                issues["potential_cuda_issues"].append((i, message))
        
        # Check for visualization without proper tensor conversion
        for pattern, message in visualization_patterns:
            for line_num, line in enumerate(cell.source.split('\n')):
                if is_visualization_context(line) and re.search(pattern, line):
                    issues["tensor_manipulation_issues"].append((i, line_num, message))
        
        # Check for functions that should have detach/cpu/numpy
        # Simplified heuristic - might have false positives
        lines = cell.source.split('\n')
        for j, line in enumerate(lines):
            if is_visualization_context(line) and 'torch' in line:
                if not re.search(r'\.detach\(\)\.cpu\(\)\.numpy\(\)', line):
                    # Skip if it's already been identified by another check
                    if not any((i, j, msg) in issues["tensor_manipulation_issues"] for msg in ["Possible tensor visualization without detach().cpu().numpy()"]):
                        issues["missing_detach_cpu"].append((i, j, line.strip()))
    
    return issues


def validate_variable_dependencies(notebook_path: str) -> Dict[str, Any]:
    """Check for variable dependencies between cells"""
    print(f"Validating variable dependencies: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    
    issues = {
        "potential_undefined_variables": [],
        "overwritten_variables": [],
        "important_variables": {}
    }
    
    # Track defined variables
    defined_variables = set()
    var_definition_cells = {}
    
    # First pass: collect variable definitions
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
            
        # Find variable assignments
        assignment_pattern = r'^([a-zA-Z][a-zA-Z0-9_]*)\s*='
        for line in cell.source.split('\n'):
            match = re.match(assignment_pattern, line.strip())
            if match:
                var_name = match.group(1)
                defined_variables.add(var_name)
                var_definition_cells[var_name] = i
    
    # Second pass: check for undefined variables
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
            
        # Check variable usage
        # This is a simplified approach and will have false positives
        var_pattern = r'\b([a-zA-Z][a-zA-Z0-9_]*)\b'
        cell_text = re.sub(r'[\'"][^\'"]*[\'"]', '', cell.source)  # Remove string literals
        for j, line in enumerate(cell_text.split('\n')):
            # Skip comments
            if line.strip().startswith('#'):
                continue
                
            # Skip function/class definition lines
            if line.strip().startswith(('def ', 'class ')):
                continue
            
            # Identify potential variable uses
            # This approach is simplistic and will have false positives
            if '=' in line:
                # For lines with assignment, only check right side
                right_side = line.split('=', 1)[1]
                matches = re.findall(var_pattern, right_side)
            else:
                matches = re.findall(var_pattern, line)
                
            for var in matches:
                # Skip common keywords, built-ins, and imports
                if var in ['for', 'if', 'else', 'elif', 'while', 'try', 'except', 'with', 
                          'True', 'False', 'None', 'import', 'from', 'as', 'return',
                          'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
                          'np', 'plt', 'pd', 'torch', 'keras', 'tf', 'model', 'ax', 'os', 'sys']:
                    continue
                    
                # Check if variable might be undefined
                if var not in defined_variables:
                    issues["potential_undefined_variables"].append((i, j, var))
                # Check if variable is used before its definition
                elif var_definition_cells[var] > i:
                    issues["potential_undefined_variables"].append((i, j, var))
    
    # Identify important variables (defined in one cell, used in many others)
    var_usage_count = {var: 0 for var in defined_variables}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
            
        for var in defined_variables:
            if var in cell.source and i != var_definition_cells.get(var, -1):
                var_usage_count[var] += 1
    
    # Variables used in 3+ cells are considered "important"
    for var, count in var_usage_count.items():
        if count >= 3:
            issues["important_variables"][var] = (var_definition_cells.get(var, -1), count)
    
    return issues


def validate_device_handling(notebook_path: str) -> Dict[str, Any]:
    """Check for proper device handling in PyTorch notebooks"""
    print(f"Validating device handling: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    
    issues = {
        "device_inconsistencies": [],
        "hardcoded_device": [],
        "missing_to_device": []
    }
    
    has_device_detection = False
    device_var_name = None
    models = []
    
    # First pass: identify device setup
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        
        # Check for device detection
        if "torch.device" in cell.source and ("cuda" in cell.source and "available" in cell.source):
            has_device_detection = True
            
            # Try to identify the device variable name
            device_pattern = r'([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*torch\.device\('
            match = re.search(device_pattern, cell.source)
            if match:
                device_var_name = match.group(1)
                
        # Identify model variables (simplistic)
        model_pattern = r'([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*(?:AutoModel|nn\.Module|AutoModelForCausalLM|from_pretrained)'
        for match in re.finditer(model_pattern, cell.source):
            models.append(match.group(1))
            
    # Second pass: check for device handling issues
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
            
        # Check for hardcoded device
        if re.search(r'\.to\([\'"]cuda[\'"]\)', cell.source) or re.search(r'device\s*=\s*[\'"]cuda[\'"]', cell.source):
            issues["hardcoded_device"].append(i)
            
        # Check for missing device transfer for model initialization
        for model in models:
            if f"{model} =" in cell.source and not f"{model}.to(" in cell.source:
                # Check if device transfer happens in the same cell after model creation
                if not re.search(f"{model}.*to\(", cell.source, re.DOTALL):
                    issues["missing_to_device"].append((i, model))
        
        # Check for inconsistent device usage
        if device_var_name:
            if re.search(r'\.to\([\'"]cuda[\'"]\)', cell.source) and device_var_name in cell.source:
                issues["device_inconsistencies"].append((i, f"Mixed usage of hardcoded 'cuda' and {device_var_name}"))
    
    return issues


def check_common_colab_issues(notebook_path: str) -> Dict[str, Any]:
    """Check for common Colab-specific issues"""
    print(f"Checking Colab compatibility: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    
    issues = {
        "file_path_issues": [],
        "missing_repo_clone": True,
        "missing_system_dependency_installation": True,
        "missing_pip_installs": True,
        "tqdm_without_colab": [],
        "matplotlib_display_issues": [],
        "interactive_widget_issues": []
    }
    
    # Keep track of the existence of key Colab components
    has_repo_clone = False
    has_system_deps = False
    has_pip_installs = False
    tqdm_imports = []
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
            
        # Check for repository cloning
        if "git clone" in cell.source:
            has_repo_clone = True
            
        # Check for system dependency installation
        if "apt-get" in cell.source or "apt install" in cell.source:
            has_system_deps = True
            
        # Check for pip installations
        if "pip install" in cell.source:
            has_pip_installs = True
            
        # Check for absolute file paths that won't work in Colab
        if re.search(r'[\'"]\/(?!content|tmp|usr|bin|lib|var|etc|dev)[a-zA-Z0-9_\.\/]+[\'"]', cell.source):
            issues["file_path_issues"].append(i)
            
        # Check for tqdm usage
        if "tqdm" in cell.source:
            tqdm_imports.append(i)
            if "tqdm.notebook" not in cell.source and "tqdm_notebook" not in cell.source:
                issues["tqdm_without_colab"].append(i)
                
        # Check for matplotlib display configuration
        if "plt." in cell.source and "%matplotlib inline" not in cell.source:
            has_correct_config = False
            for c in nb.cells:
                if c.cell_type == "code" and "%matplotlib inline" in c.source:
                    has_correct_config = True
                    break
            if not has_correct_config:
                issues["matplotlib_display_issues"].append(i)
                
        # Check for interactive widgets without proper Colab setup
        if "ipywidgets" in cell.source or "widgets." in cell.source:
            has_widgets_setup = False
            for c in nb.cells:
                if c.cell_type == "code" and "ipywidgets" in c.source and "import ipywidgets" in c.source:
                    has_widgets_setup = True
                    break
            if not has_widgets_setup:
                issues["interactive_widget_issues"].append(i)
    
    # Update findings
    issues["missing_repo_clone"] = not has_repo_clone
    issues["missing_system_dependency_installation"] = not has_system_deps
    issues["missing_pip_installs"] = not has_pip_installs
    
    return issues


def validate_error_handling(notebook_path: str) -> Dict[str, Any]:
    """Check for proper error handling in the notebook"""
    print(f"Validating error handling: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    
    issues = {
        "missing_try_except": [],
        "potentially_dangerous_operations": [],
        "good_error_handling": []
    }
    
    # Operations that should have error handling
    dangerous_operations = [
        (r'open\(', "File operations"),
        (r'os\.remove\(|os\.unlink\(', "File deletion"),
        (r'shutil\.rmtree\(', "Directory deletion"),
        (r'subprocess\.(?:call|run|Popen)', "Subprocess execution"),
        (r'requests\.(?:get|post|put|delete)', "HTTP requests")
    ]
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
            
        # Check for operations that should have error handling
        for pattern, description in dangerous_operations:
            if re.search(pattern, cell.source):
                # Check if there's a try-except block
                if "try:" in cell.source and "except" in cell.source:
                    issues["good_error_handling"].append((i, description))
                else:
                    issues["potentially_dangerous_operations"].append((i, description))
                    
        # Check for long cells that should have error handling
        if len(cell.source.strip().split('\n')) > 20 and "try:" not in cell.source:
            has_risky_ops = False
            for risky_op in ['os.', 'shutil.', 'open(', 'with open', '.fit(', '.save(', '.load(', 
                            'download', 'requests.', 'subprocess.']:
                if risky_op in cell.source:
                    has_risky_ops = True
                    break
            if has_risky_ops:
                issues["missing_try_except"].append(i)
    
    return issues


def comprehensive_notebook_validation(notebook_path: str) -> Dict[str, Any]:
    """Run all validation checks on a notebook"""
    if not os.path.exists(notebook_path):
        return {"error": f"Notebook not found: {notebook_path}"}
        
    print(f"Starting comprehensive validation of: {notebook_path}")
    
    # Run all validation checks
    results = {
        "execution_order": validate_cell_execution_order(notebook_path),
        "imports": validate_import_statements(notebook_path),
        "tensor_operations": validate_tensor_operations(notebook_path),
        "variable_dependencies": validate_variable_dependencies(notebook_path),
        "device_handling": validate_device_handling(notebook_path),
        "colab_issues": check_common_colab_issues(notebook_path),
        "error_handling": validate_error_handling(notebook_path)
    }
    
    # Print summary
    print("\n===== VALIDATION SUMMARY =====")
    
    total_issues = 0
    for category, checks in results.items():
        category_issues = 0
        
        print(f"\n{category.upper()} CHECKS:")
        for check_name, check_result in checks.items():
            # Different handling based on result type
            if isinstance(check_result, list):
                if check_result:
                    print(f"  ❌ {check_name}: {len(check_result)} issues")
                    category_issues += len(check_result)
                else:
                    print(f"  ✅ {check_name}: No issues")
            elif isinstance(check_result, dict):
                if check_result:
                    print(f"  ❌ {check_name}: {len(check_result)} issues")
                    category_issues += len(check_result)
                else:
                    print(f"  ✅ {check_name}: No issues")
            elif isinstance(check_result, set):
                if check_result:
                    print(f"  ❌ {check_name}: {len(check_result)} issues")
                    category_issues += len(check_result)
                else:
                    print(f"  ✅ {check_name}: No issues")
            elif isinstance(check_result, bool):
                if check_result:
                    print(f"  ❌ {check_name}")
                    category_issues += 1
                else:
                    print(f"  ✅ {check_name}: No issues")
            elif isinstance(check_result, (int, float)):
                if check_result > 0:
                    print(f"  ❌ {check_name}: {check_result}")
                    category_issues += check_result
                else:
                    print(f"  ✅ {check_name}: No issues")
        
        if category_issues == 0:
            print(f"All {category} checks passed!")
        else:
            print(f"Found {category_issues} issues in {category} checks")
            total_issues += category_issues
    
    print(f"\nTOTAL ISSUES: {total_issues}")
    if total_issues == 0:
        print("✅ Notebook looks good! No issues found.")
    else:
        print("❌ Notebook has issues that should be addressed.")
    
    return results


if __name__ == "__main__":
    # Get notebook path from command line or use default
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else "colab_notebooks/NeuralPlasticityDemo.ipynb"
    
    # Run comprehensive validation
    comprehensive_notebook_validation(notebook_path)