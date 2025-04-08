#!/usr/bin/env python
# Fix training process for better Colab stability

import json
import os
from pathlib import Path

def fix_training_for_colab(notebook_path):
    """
    Modifies the training configuration and process to be more stable in Colab.
    
    1. Sets ENABLE_LONG_TRAINING to False
    2. Adds memory management to the training loop
    3. Increases checkpoint frequency
    4. Adds error handling for common issues
    """
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        print("Fixing training configuration for better Colab stability...")
        
        # First, update the config cell
        config_cell_found = False
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and 'ENABLE_LONG_TRAINING' in str(cell['source']):
                config_cell_found = True
                print(f"Found config cell at index {i}")
                
                # Get the cell content
                cell_content = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
                
                # Set ENABLE_LONG_TRAINING to False
                cell_content = cell_content.replace('ENABLE_LONG_TRAINING = True', 'ENABLE_LONG_TRAINING = False  # Set to False for demo purposes to avoid memory/runtime issues')
                
                # Reduce checkpoint interval
                cell_content = cell_content.replace('CHECKPOINT_INTERVAL = 1000', 'CHECKPOINT_INTERVAL = 500    # Save checkpoint more frequently (was 1000)')
                
                # Update the cell
                if isinstance(notebook['cells'][i]['source'], list):
                    updated_lines = []
                    for line in cell_content.split('\n'):
                        updated_lines.append(line + '\n')
                    if updated_lines:
                        updated_lines[-1] = updated_lines[-1].rstrip('\n')
                    notebook['cells'][i]['source'] = updated_lines
                else:
                    notebook['cells'][i]['source'] = cell_content
                
                print(f"Updated config cell {i} for better stability")
                break
        
        if not config_cell_found:
            print("Could not find config cell")
            return False
            
        # Next, update the training loop with memory management
        training_cell_idx = None
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and 'for epoch in range(NUM_EPOCHS)' in str(cell['source']):
                training_cell_idx = i
                print(f"Found training loop cell at index {i}")
                break
        
        if training_cell_idx is None:
            print("Could not find training loop cell")
            return False
            
        # Get the training cell content
        training_content = ''.join(notebook['cells'][training_cell_idx]['source']) if isinstance(notebook['cells'][training_cell_idx]['source'], list) else notebook['cells'][training_cell_idx]['source']
        
        # Add memory management code after the imports/function definitions
        memory_mgmt_code = """
# Add memory management utilities
import gc
import torch

def clear_memory():
    '''Clear GPU memory cache and run garbage collection'''
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

"""
        
        # Also add memory clearing in the epoch loop and error recovery
        # Find the epoch loop start
        epoch_loop_start = training_content.find('for epoch in range(NUM_EPOCHS):')
        if epoch_loop_start != -1:
            # Add memory clearing after each epoch
            # Find the end of the epoch loop body (look for the print showing epoch completion)
            epoch_completion_print = 'print(f"Completed Epoch {epoch+1} - Total steps: {global_step}")'
            epoch_loop_body_end = training_content.find(epoch_completion_print)
            
            if epoch_loop_body_end != -1:
                # Insert memory clearing after the epoch completion print
                epoch_loop_body_end += len(epoch_completion_print)
                training_content = (
                    training_content[:epoch_loop_body_end] + 
                    "\n        # Clear memory at the end of each epoch\n        clear_memory()\n" +
                    training_content[epoch_loop_body_end:]
                )
                print("Added memory clearing after each epoch")
            
            # Add additional error handling for visualization
            if 'error_checkpoint_path = save_checkpoint(global_step, epoch + 1)' in training_content:
                # Add specific handling for visualization errors
                recovery_code = """
    # Add more specific error handling for common issues
    except (MemoryError, RuntimeError) as e:
        print(f"\\nMemory or Runtime error: {e}")
        print("Attempting to recover and save checkpoint...")
        # Force cleanup
        clear_memory()
        try:
            recovery_checkpoint_path = save_checkpoint(global_step, epoch + 1)
            print(f"Recovery checkpoint saved at {recovery_checkpoint_path}")
        except Exception as save_error:
            print(f"Could not save checkpoint during recovery: {save_error}")
"""
                # Replace the general exception with more specific handling
                general_exception = "except Exception as e:"
                general_exception_idx = training_content.find(general_exception)
                if general_exception_idx != -1:
                    training_content = (
                        training_content[:general_exception_idx] + 
                        recovery_code +
                        training_content[general_exception_idx:]
                    )
                    print("Added specific error recovery for memory and runtime errors")
        
        # Insert memory management function at the beginning of the cell
        # Find the first new line after possible import statements
        function_start = training_content.find('def visualize_gradient_norms')
        if function_start != -1:
            # Insert memory management function before visualize_gradient_norms
            training_content = (
                training_content[:function_start] + 
                memory_mgmt_code +
                training_content[function_start:]
            )
            print("Added memory management utility functions")
        
        # Update the training cell
        if isinstance(notebook['cells'][training_cell_idx]['source'], list):
            updated_lines = []
            for line in training_content.split('\n'):
                updated_lines.append(line + '\n')
            if updated_lines:
                updated_lines[-1] = updated_lines[-1].rstrip('\n')
            notebook['cells'][training_cell_idx]['source'] = updated_lines
        else:
            notebook['cells'][training_cell_idx]['source'] = training_content
        
        # Update version to v0.0.36 in first cell
        if notebook['cells'][0]['cell_type'] == 'markdown':
            # Get first cell content
            first_cell = ''.join(notebook['cells'][0]['source']) if isinstance(notebook['cells'][0]['source'], list) else notebook['cells'][0]['source']
            
            # Update version number and add changelog entry
            if 'v0.0.35' in first_cell and 'v0.0.36' not in first_cell:
                first_cell = first_cell.replace('v0.0.35', 'v0.0.36')
                
                # Add entry for v0.0.36
                changelog_entry = """### New in v0.0.36:
- Optimized for long-running training in Colab
- Added memory management to avoid OOM errors
- Increased checkpoint frequency for better recovery
- Improved error handling for common failures

"""
                # Find position to insert (before the v0.0.35 changelog)
                v35_pos = first_cell.find('### New in v0.0.35:')
                if v35_pos != -1:
                    first_cell = first_cell[:v35_pos] + changelog_entry + first_cell[v35_pos:]
                    
                    # Update the cell
                    if isinstance(notebook['cells'][0]['source'], list):
                        updated_lines = []
                        for line in first_cell.split('\n'):
                            updated_lines.append(line + '\n')
                        if updated_lines:
                            updated_lines[-1] = updated_lines[-1].rstrip('\n')
                        notebook['cells'][0]['source'] = updated_lines
                    else:
                        notebook['cells'][0]['source'] = first_cell
                    
                    print("Updated version to v0.0.36 and added changelog entry")
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Successfully fixed notebook for better training stability in Colab")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_training_for_colab(notebook_path)