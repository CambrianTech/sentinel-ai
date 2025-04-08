#!/usr/bin/env python
# Fix for entropy calculation that addresses numerical stability issues

import torch
import re
import json
from pathlib import Path

def improved_entropy_calculation(attention_pattern):
    """
    Calculate entropy of attention patterns with better numerical stability.
    
    Args:
        attention_pattern: Attention pattern tensor of shape [..., seq_len]
        
    Returns:
        Entropy tensor with last dimension reduced
    """
    # Add small epsilon to avoid numerical issues with log(0)
    epsilon = 1e-8
    
    # Ensure attention is positive
    attention_pattern = attention_pattern.clamp(min=epsilon)
    
    # Normalize along sequence dimension to ensure it sums to 1
    # This is critical for entropy calculation
    norm_attention = attention_pattern / attention_pattern.sum(dim=-1, keepdim=True)
    
    # Calculate entropy: -sum(p * log(p))
    log_probs = torch.log(norm_attention)
    entropy = -torch.sum(norm_attention * log_probs, dim=-1)
    
    return entropy

def fix_entropy_calculation_in_notebook(notebook_path, output_path=None):
    """
    Fix the entropy calculation in the notebook to prevent zero entropy values.
    
    Args:
        notebook_path: Path to the notebook file
        output_path: Path to save the modified notebook (if None, overwrite original)
    """
    if output_path is None:
        output_path = notebook_path
    
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the debug cell where entropy is being calculated
    debug_cell_found = False
    controller_cell_found = False
    
    # First try to find the controller.collect_head_metrics call
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Look for the debug section where entropy is collected
            if 'debug_entropy, debug_grads = controller.collect_head_metrics(' in source:
                debug_cell_found = True
                print(f"Found debug entropy collection in cell {i}")
                
                # Look for the place after entropy statistics are printed
                lines = cell['source']
                modified_lines = []
                
                for j, line in enumerate(lines):
                    modified_lines.append(line)
                    
                    # Add our improved entropy calculation after the debugging section
                    if "Non-zero values:" in line and "entropy" in line:
                        # Add the improved entropy calculation function
                        modified_lines.append("\n# Add improved entropy calculation to fix zero entropy values\n")
                        modified_lines.append("def improved_entropy_calculation(attention_pattern):\n")
                        modified_lines.append("    \"\"\"\n")
                        modified_lines.append("    Calculate entropy of attention patterns with better numerical stability.\n")
                        modified_lines.append("    \"\"\"\n")
                        modified_lines.append("    # Add small epsilon to avoid numerical issues with log(0)\n")
                        modified_lines.append("    epsilon = 1e-8\n")
                        modified_lines.append("    \n")
                        modified_lines.append("    # Ensure attention is positive\n")
                        modified_lines.append("    attention_pattern = attention_pattern.clamp(min=epsilon)\n")
                        modified_lines.append("    \n")
                        modified_lines.append("    # Normalize along sequence dimension to ensure it sums to 1\n")
                        modified_lines.append("    norm_attention = attention_pattern / attention_pattern.sum(dim=-1, keepdim=True)\n")
                        modified_lines.append("    \n")
                        modified_lines.append("    # Calculate entropy: -sum(p * log(p))\n")
                        modified_lines.append("    log_probs = torch.log(norm_attention)\n")
                        modified_lines.append("    entropy = -torch.sum(norm_attention * log_probs, dim=-1)\n")
                        modified_lines.append("    \n")
                        modified_lines.append("    return entropy\n\n")
                        
                        # Add improved debugging with the fixed entropy calculation
                        modified_lines.append("# Try the improved entropy calculation function\n")
                        modified_lines.append("print(\"\\nTrying improved entropy calculation...\")\n")
                        modified_lines.append("try:\n")
                        modified_lines.append("    # Get attention outputs\n")
                        modified_lines.append("    inputs = next(iter(validation_dataloader))\n")
                        modified_lines.append("    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}\n")
                        modified_lines.append("    \n")
                        modified_lines.append("    model.eval()\n")
                        modified_lines.append("    with torch.no_grad():\n")
                        modified_lines.append("        outputs = model(**inputs, output_attentions=True)\n")
                        modified_lines.append("    \n")
                        modified_lines.append("    if hasattr(outputs, 'attentions') and outputs.attentions is not None:\n")
                        modified_lines.append("        attn = outputs.attentions[0]  # First layer's attention\n")
                        modified_lines.append("        # Calculate entropy using improved function\n")
                        modified_lines.append("        fixed_entropy = improved_entropy_calculation(attn)\n")
                        modified_lines.append("        \n")
                        modified_lines.append("        print(f\"Improved entropy calculation results:\")\n")
                        modified_lines.append("        print(f\"Mean entropy: {fixed_entropy.mean().item():.4f}\")\n")
                        modified_lines.append("        print(f\"Min entropy: {fixed_entropy.min().item():.4f}\")\n")
                        modified_lines.append("        print(f\"Max entropy: {fixed_entropy.max().item():.4f}\")\n")
                        modified_lines.append("        print(f\"Are all entropy values the same? {torch.allclose(fixed_entropy, fixed_entropy[0,0])}\")\n")
                        modified_lines.append("        print(f\"Non-zero values: {torch.count_nonzero(fixed_entropy)}/{fixed_entropy.numel()}\")\n")
                        modified_lines.append("except Exception as e:\n")
                        modified_lines.append("    print(f\"Error testing improved entropy calculation: {e}\")\n")
                
                # Update the cell
                cell['source'] = modified_lines
            
            # Find the controller creation cell to patch the internal entropy calculation
            if "controller = create_plasticity_controller(" in source:
                controller_cell_found = True
                print(f"Found controller creation in cell {i}")
                
                # Add monkey patching code after controller creation
                lines = cell['source']
                modified_lines = []
                
                for line in lines:
                    modified_lines.append(line)
                    
                    # Add monkey patching after controller creation
                    if "controller = create_plasticity_controller(" in line and ")" in line:
                        modified_lines.append("\n# Monkey patch the controller's entropy calculation to use our improved version\n")
                        modified_lines.append("# Only do this if we haven't already defined the improved_entropy_calculation function\n")
                        modified_lines.append("if 'improved_entropy_calculation' not in globals():\n")
                        modified_lines.append("    def improved_entropy_calculation(attention_pattern):\n")
                        modified_lines.append("        \"\"\"\n")
                        modified_lines.append("        Calculate entropy of attention patterns with better numerical stability.\n")
                        modified_lines.append("        \"\"\"\n")
                        modified_lines.append("        # Add small epsilon to avoid numerical issues with log(0)\n")
                        modified_lines.append("        epsilon = 1e-8\n")
                        modified_lines.append("        \n")
                        modified_lines.append("        # Ensure attention is positive\n")
                        modified_lines.append("        attention_pattern = attention_pattern.clamp(min=epsilon)\n")
                        modified_lines.append("        \n")
                        modified_lines.append("        # Normalize along sequence dimension to ensure it sums to 1\n")
                        modified_lines.append("        norm_attention = attention_pattern / attention_pattern.sum(dim=-1, keepdim=True)\n")
                        modified_lines.append("        \n")
                        modified_lines.append("        # Calculate entropy: -sum(p * log(p))\n")
                        modified_lines.append("        log_probs = torch.log(norm_attention)\n")
                        modified_lines.append("        entropy = -torch.sum(norm_attention * log_probs, dim=-1)\n")
                        modified_lines.append("        \n")
                        modified_lines.append("        return entropy\n\n")
                        
                        # Add code to patch the controller object
                        modified_lines.append("# Replace the controller's entropy calculation method\n")
                        modified_lines.append("import types\n")
                        modified_lines.append("def patched_calculate_attention_entropy(self, attention_weights):\n")
                        modified_lines.append("    # Use our improved entropy calculation\n")
                        modified_lines.append("    return improved_entropy_calculation(attention_weights)\n")
                        modified_lines.append("\n")
                        modified_lines.append("# Apply the patch to the controller instance\n")
                        modified_lines.append("if hasattr(controller, 'calculate_attention_entropy'):\n")
                        modified_lines.append("    controller.calculate_attention_entropy = types.MethodType(patched_calculate_attention_entropy, controller)\n")
                        modified_lines.append("    print(\"Applied entropy calculation patch to controller\")\n")
                
                # Update the cell
                cell['source'] = modified_lines
    
    # Update version history and changelog
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and '# Neural Plasticity Demo' in ''.join(cell['source']):
            source = ''.join(cell['source'])
            version_pattern = r'v0\.0\.(\d+)'
            match = re.search(version_pattern, source)
            if match:
                current_version = int(match.group(1))
                new_version = current_version + 1
                
                # Update version number
                source = re.sub(
                    version_pattern, 
                    f'v0.0.{new_version}',
                    source
                )
                
                # Add changelog entry if not already there
                if 'Fixed entropy calculation' not in source:
                    new_in_pattern = r'(### New in v0\.0\.\d+:[\s\S]*?)(?=###|$)'
                    match = re.search(new_in_pattern, source)
                    if match:
                        changelog_entry = (
                            f"### New in v0.0.{new_version}:\n"
                            "- Fixed entropy calculation to prevent zero values\n"
                            "- Added numerical stability improvements\n"
                            "- Properly normalized attention patterns\n"
                        )
                        source = re.sub(
                            new_in_pattern,
                            changelog_entry + match.group(1),
                            source
                        )
                
                notebook['cells'][i]['source'] = source.split('\n')
                break
    
    # Update conclusion cell with version history
    for i, cell in enumerate(notebook['cells']):
        if (cell['cell_type'] == 'markdown' and 
            '## Conclusion' in ''.join(cell['source']) and 
            '## Version History' in ''.join(cell['source'])):
            
            source = ''.join(cell['source'])
            # Find the version history section
            version_history_pattern = r'(## Version History\n\n)([\s\S]*)'
            match = re.search(version_history_pattern, source)
            if match:
                version_prefix = match.group(1)
                existing_history = match.group(2)
                
                # Add new version entry at the top
                new_version_entry = (
                    f"- v0.0.{new_version}: Fixed entropy calculation to prevent zero values, "
                    "added numerical stability improvements, properly normalized attention patterns\n\n"
                )
                
                source = re.sub(
                    version_history_pattern,
                    version_prefix + new_version_entry + existing_history,
                    source
                )
                
                notebook['cells'][i]['source'] = source.split('\n')
                break
    
    # Save the modified notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    if debug_cell_found or controller_cell_found:
        print(f"Successfully modified notebook: {output_path}")
        if debug_cell_found:
            print("- Added improved entropy calculation in debug cell")
        if controller_cell_found:
            print("- Patched controller's entropy calculation method")
        return True
    else:
        print(f"Warning: Could not find appropriate cells to modify in {notebook_path}")
        return False

if __name__ == "__main__":
    # Locate the notebook
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    
    if notebook_path.exists():
        # Apply the fix
        fix_entropy_calculation_in_notebook(notebook_path)
        print("Entropy calculation fix applied to the notebook.")
        print("To apply the changes, re-run the notebook from after the cell where entropy is calculated.")
    else:
        print(f"Notebook not found at: {notebook_path}")