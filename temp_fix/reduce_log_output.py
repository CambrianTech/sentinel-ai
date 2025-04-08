#!/usr/bin/env python
# Reduce log output in NeuralPlasticityDemo.ipynb

import json
import os
from pathlib import Path

def reduce_log_output(notebook_path):
    """
    Reduces excessive log output during training in the notebook.
    
    1. Adds tracking of previous state to reduce redundant logs
    2. Reduces verbosity in pruning functions
    3. Only shows metrics when pruning status changes
    4. Implements smart visualization to avoid redundant displays
    """
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        print("Reducing log output during training...")
        
        # First, update the pruning function
        pruning_cell_idx = None
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and 'def apply_gradient_pruning' in str(cell['source']):
                pruning_cell_idx = i
                print(f"Found pruning function at cell {i}")
                break
        
        if pruning_cell_idx is not None:
            cell_content = ''.join(notebook['cells'][pruning_cell_idx]['source']) if isinstance(notebook['cells'][pruning_cell_idx]['source'], list) else notebook['cells'][pruning_cell_idx]['source']
            
            # Reduce verbosity in pruning function
            cell_content = cell_content.replace('verbose=True', 'verbose=False  # Reduce verbosity')
            
            # Add conditional log output based on first run
            if "Print stats about the pruned and kept heads" in cell_content:
                # Find where the print statements start
                print_start = cell_content.find("# Print stats about the pruned and kept heads")
                if print_start != -1:
                    # Get the rest of the function after this comment
                    print_section = cell_content[print_start:]
                    # Find where the return statement is
                    return_start = print_section.find("return pruned_heads")
                    if return_start != -1:
                        # Get the print section up to the return
                        print_content = print_section[:return_start]
                        
                        # Create a new version with conditional output
                        new_print_content = """# Print stats about the pruned and kept heads - but only if we actually pruned something
    if pruned_heads:
        print(f"Pruned {len(pruned_heads)} heads with lowest gradient norms")
        # Only show detailed metrics at the start
        if not hasattr(apply_gradient_pruning, "has_pruned_before"):
            avg_pruned = grad_norm_values[pruning_mask].mean().item()
            avg_kept = grad_norm_values[~pruning_mask].mean().item()
            print(f"Average gradient of pruned heads: {avg_pruned:.6f}")
            print(f"Average gradient of kept heads: {avg_kept:.6f}")
            print(f"Ratio (kept/pruned): {avg_kept/avg_pruned:.2f}x")
            # Set flag to avoid showing these details every time
            apply_gradient_pruning.has_pruned_before = True
    """
                        
                        # Replace the original print section
                        cell_content = cell_content[:print_start] + new_print_content + print_section[return_start:]
            
            # Update the cell
            if isinstance(notebook['cells'][pruning_cell_idx]['source'], list):
                lines = cell_content.split('\n')
                updated_lines = [line + '\n' for line in lines]
                if updated_lines:
                    updated_lines[-1] = updated_lines[-1].rstrip('\n')
                notebook['cells'][pruning_cell_idx]['source'] = updated_lines
            else:
                notebook['cells'][pruning_cell_idx]['source'] = cell_content
            
            print(f"Reduced verbosity in pruning function")
        
        # Next, update the training loop
        training_cell_idx = None
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and 'for epoch in range(NUM_EPOCHS)' in str(cell['source']):
                training_cell_idx = i
                print(f"Found training loop at cell {i}")
                break
        
        if training_cell_idx is not None:
            cell_content = ''.join(notebook['cells'][training_cell_idx]['source']) if isinstance(notebook['cells'][training_cell_idx]['source'], list) else notebook['cells'][training_cell_idx]['source']
            
            # Add tracking variables to reduce logging
            if "try:" in cell_content and "for epoch in range(NUM_EPOCHS):" in cell_content:
                # Insert state tracking variables after the try statement
                try_pos = cell_content.find("try:")
                try_line_end = cell_content.find("\n", try_pos)
                if try_pos != -1 and try_line_end != -1:
                    tracking_code = """
    # Track previous state to reduce logging
    last_total_pruned = 0
    last_visualization_step = 0
"""
                    cell_content = cell_content[:try_line_end+1] + tracking_code + cell_content[try_line_end+1:]
            
            # Modify the print statements to only show when something changes
            if "Print status with epoch information" in cell_content:
                # Find where status printing starts
                status_print_start = cell_content.find("# Print status with epoch information")
                if status_print_start != -1:
                    # Find where the next part begins after printing (could be visualization)
                    next_section_start = cell_content.find("# Generate and save the visualization", status_print_start)
                    if next_section_start != -1:
                        # Get the print section
                        print_section = cell_content[status_print_start:next_section_start]
                        
                        # Create a new version with reduced output
                        new_print_section = """# Print status with epoch information - reduced output
                print(f"  Step {global_step} (Epoch {epoch+1}) - Train loss: {epoch_loss / epoch_steps:.4f}, "
                      f"Eval loss: {eval_loss:.4f}, Perplexity: {eval_perplexity:.2f}")
                
                # Only print pruning details if something changed
                if total_pruned != last_total_pruned:
                    print(f"  Pruned: {len(pruned_heads)} heads, Revived: {len(revived_heads)} heads, Total pruned: {total_pruned}")
                    print(f"  Sparsity: {model_info['sparsity']:.4f}")
                    last_total_pruned = total_pruned
                """
                        
                        # Replace the original print section
                        cell_content = cell_content[:status_print_start] + new_print_section + cell_content[next_section_start:]
            
            # Modify the visualization to be smarter
            if "Generate and save the visualization" in cell_content:
                # Find where visualization code starts
                viz_start = cell_content.find("# Generate and save the visualization")
                if viz_start != -1:
                    # Find where the visualization code block ends
                    viz_end = cell_content.find("# Run model inference at regular intervals", viz_start)
                    if viz_end != -1:
                        # Get the visualization section
                        viz_section = cell_content[viz_start:viz_end]
                        
                        # Find the if statement that controls visualization
                        if_start = viz_section.find("if len(pruned_heads) > 0 or global_step % VISUALIZATION_INTERVAL == 0:")
                        if if_start != -1:
                            # Get the part before the if statement
                            viz_before_if = viz_section[:if_start]
                            
                            # Get the part inside the if block
                            if_block_start = viz_section.find(":", if_start)
                            if_block = viz_section[if_block_start+1:]
                            
                            # Create a new smarter if condition
                            new_if_condition = """# Generate and save the visualization with pruning overlays if:
                # 1. New heads were pruned, OR
                # 2. At regular visualization intervals but not too frequently
                should_visualize = (
                    len(pruned_heads) > 0 or 
                    (global_step % VISUALIZATION_INTERVAL == 0 and 
                     global_step - last_visualization_step >= VISUALIZATION_INTERVAL)
                )
                
                if should_visualize:
                    last_visualization_step = global_step"""
                            
                            # Replace the original if condition
                            new_viz_section = viz_before_if + new_if_condition + if_block
                            
                            # Replace the original visualization section
                            cell_content = cell_content[:viz_start] + new_viz_section + cell_content[viz_end:]
            
            # Update the cell
            if isinstance(notebook['cells'][training_cell_idx]['source'], list):
                lines = cell_content.split('\n')
                updated_lines = [line + '\n' for line in lines]
                if updated_lines:
                    updated_lines[-1] = updated_lines[-1].rstrip('\n')
                notebook['cells'][training_cell_idx]['source'] = updated_lines
            else:
                notebook['cells'][training_cell_idx]['source'] = cell_content
            
            print(f"Optimized log output in training loop")
        
        # Update version number to v0.0.37
        if notebook['cells'][0]['cell_type'] == 'markdown':
            first_cell = ''.join(notebook['cells'][0]['source']) if isinstance(notebook['cells'][0]['source'], list) else notebook['cells'][0]['source']
            
            # Update version number
            if 'v0.0.36' in first_cell and 'v0.0.37' not in first_cell:
                first_cell = first_cell.replace('v0.0.36', 'v0.0.37')
                
                # Add changelog entry
                changelog_entry = """### New in v0.0.37:
- Reduced log output during training
- Only show metrics when pruning status changes
- Added smart visualization to reduce redundant displays
- Decreased verbosity in pruning functions

"""
                # Find where to insert the changelog entry
                v36_pos = first_cell.find('### New in v0.0.36:')
                if v36_pos != -1:
                    first_cell = first_cell[:v36_pos] + changelog_entry + first_cell[v36_pos:]
                    
                    # Update the cell
                    if isinstance(notebook['cells'][0]['source'], list):
                        lines = first_cell.split('\n')
                        updated_lines = [line + '\n' for line in lines]
                        if updated_lines:
                            updated_lines[-1] = updated_lines[-1].rstrip('\n')
                        notebook['cells'][0]['source'] = updated_lines
                    else:
                        notebook['cells'][0]['source'] = first_cell
                    
                    print("Updated version to v0.0.37 and added changelog entry")
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Successfully reduced log output in NeuralPlasticityDemo.ipynb")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    reduce_log_output(notebook_path)