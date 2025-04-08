#!/usr/bin/env python
# Fixed script to fix attention visualization in NeuralPlasticityDemo.ipynb

import json
import os
from pathlib import Path

def fix_attention_visualization(notebook_path):
    """
    Manually fixes attention visualizations in the notebook to ensure proper scaling.
    
    This addresses issues where attention heatmaps appear broken or improperly scaled.
    The fix updates cells 17, 18, and 20 to ensure proper visualization.
    """
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        print("Fixing attention visualization in specific cells...")
        
        # Create our edited versions of the cells with fixes
        # Fix cell 17 - Add clim to attention visualization
        cell_17_source = """# Debug: Let's check the actual entropy values we're dealing with
print("\\nCollecting initial entropy and gradient metrics for debugging...")

# Make absolutely sure controller, model, and validation_dataloader are defined
try:
    # Check if we have all the necessary variables
    controller
    model
    validation_dataloader
    device
except NameError as e:
    print(f"ERROR: Missing variable: {e}")
    print("Please run the previous cells first to set up model, controller, and data.")
    # Create empty placeholders to allow this cell to run
    if 'controller' not in globals():
        print("Creating placeholder controller...")
        from types import SimpleNamespace
        controller = SimpleNamespace()
        controller.collect_head_metrics = lambda *args, **kwargs: (torch.zeros(12, 12), torch.zeros(12, 12))
    raise

# This will collect attention entropy and gradient values
try:
    debug_entropy, debug_grads = controller.collect_head_metrics(
        validation_dataloader,
        num_batches=2
    )
    
    # Print entropy statistics
    print("\\nEntropy statistics:")
    print(f"Mean entropy: {debug_entropy.mean().item():.4f}")
    print(f"Min entropy: {debug_entropy.min().item():.4f}")
    print(f"Max entropy: {debug_entropy.max().item():.4f}")
    print(f"25th percentile: {torch.quantile(debug_entropy.flatten(), 0.25).item():.4f}")
    print(f"50th percentile: {torch.quantile(debug_entropy.flatten(), 0.5).item():.4f}")
    print(f"75th percentile: {torch.quantile(debug_entropy.flatten(), 0.75).item():.4f}")
    print(f"Are all entropy values the same? {torch.allclose(debug_entropy, debug_entropy[0,0])}")
    print(f"Non-zero values: {torch.count_nonzero(debug_entropy)}/{debug_entropy.numel()}")
    
    # Add diagnostic to debug attention probability tensor
    print("\\nDIAGNOSTIC: Checking raw attention probability distributions...")
    try:
        # Get a data batch safely
        inputs = next(iter(validation_dataloader))
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        else:
            inputs = {"input_ids": inputs[0].to(device)}
        
        # Get model outputs with attention
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Analyze attention tensors
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attn_tensors = outputs.attentions
            layer_idx = 0  # Check first layer
            
            if len(attn_tensors) > 0:
                attn = attn_tensors[layer_idx]  # First layer attention
                
                # Print attention tensor stats to verify it's a valid probability distribution
                print(f"Attention tensor shape: {attn.shape}")
                print(f"Attention tensor dtype: {attn.dtype}")
                print(f"Attention tensor stats: min={attn.min().item():.6e}, max={attn.max().item():.6e}, mean={attn.mean().item():.6e}")
                
                # Check if values sum to 1 along attention dimension
                attn_sum = attn.sum(dim=-1)
                print(f"Sum along attention dimension: min={attn_sum.min().item():.6f}, max={attn_sum.max().item():.6f}")
                print(f"Close to 1.0? {torch.allclose(attn_sum, torch.ones_like(attn_sum), rtol=1e-3)}")
                
                # Check for very small values that might cause log(0) issues
                small_values = (attn < 1e-6).float().mean().item() * 100
                print(f"Percentage of very small values (<1e-6): {small_values:.2f}%")
                
                # Check for NaN or infinity
                print(f"Contains NaN: {torch.isnan(attn).any().item()}")
                print(f"Contains Inf: {torch.isinf(attn).any().item()}")
                
                # Fix entropy calculation function with better defaults
                def improved_entropy_calculation(attn_probs, eps=1e-8):
                    """Compute entropy with better numerical stability."""
                    # Ensure valid probability distribution
                    attn_probs = attn_probs.clamp(min=eps)
                    normalized_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)
                    
                    # Compute entropy
                    log_probs = torch.log(normalized_probs)
                    entropy = -torch.sum(normalized_probs * log_probs, dim=-1)
                    return entropy
                
                # Calculate entropy using improved function
                improved_entropy = improved_entropy_calculation(attn).mean(dim=(0, 1))
                print("\\nImproved entropy calculation results:")
                print(f"Mean entropy: {improved_entropy.mean().item():.4f}")
                print(f"Min entropy: {improved_entropy.min().item():.4f}")
                print(f"Max entropy: {improved_entropy.max().item():.4f}")
                
                # Add visualization of attention patterns
                print("\\nVisualizing attention pattern for one head...")
                head_idx = 0
                plt.figure(figsize=(8, 6))
                attention_map = plt.imshow(attn[0, head_idx].cpu().numpy(), cmap='viridis')
                plt.clim(0, 1.0)  # Ensure proper scale for attention visualization
                plt.colorbar(label='Attention probability')
                plt.title(f'Attention pattern (layer {layer_idx}, head {head_idx})')
                plt.xlabel('Sequence position (to)')
                plt.ylabel('Sequence position (from)')
                plt.show()
                
                # Add histogram of attention values
                plt.figure(figsize=(8, 4))
                plt.hist(attn[0, head_idx].flatten().cpu().numpy(), bins=50, alpha=0.7)
                plt.title('Histogram of attention probabilities')
                plt.xlabel('Probability value')
                plt.ylabel('Frequency')
                plt.grid(alpha=0.3)
                plt.show()
            else:
                print("No attention tensors found in the model output")
        else:
            print("Model output doesn't have attention tensors. Check if output_attentions=True is supported.")
    except Exception as e:
        print(f"Error in attention diagnostic: {e}")
        
    # Print gradient statistics
    print("\\nGradient statistics:")
    print(f"Mean gradient norm: {debug_grads.mean().item():.4f}")
    print(f"Min gradient norm: {debug_grads.min().item():.4f}")
    print(f"Max gradient norm: {debug_grads.max().item():.4f}")
    print(f"25th percentile: {torch.quantile(debug_grads.flatten(), 0.25).item():.4f}")
    print(f"50th percentile: {torch.quantile(debug_grads.flatten(), 0.5).item():.4f}")
    print(f"75th percentile: {torch.quantile(debug_grads.flatten(), 0.75).item():.4f}")
except Exception as e:
    print(f"Error collecting metrics: {e}")"""
        
        # Convert Cell 17
        cell_17_lines = []
        for line in cell_17_source.split('\n'):
            cell_17_lines.append(line + '\n')
        if cell_17_lines and cell_17_lines[-1].endswith('\n'):
            cell_17_lines[-1] = cell_17_lines[-1].rstrip('\n')
            
        # Fix cell 18 - Fix attention pattern visualization
        cell_18_source = """# Enhanced Entropy Analysis
# Run this after collecting the metrics to better understand the entropy issues

# Function to compute improved entropy with diagnostics
def compute_improved_entropy(attn_probs, eps=1e-8, debug=True):
    """Compute entropy with better numerical stability and detailed diagnostics."""
    if debug:
        # Print raw attention stats
        print(f"Raw attention shape: {attn_probs.shape}")
        print(f"Raw min/max/mean: {attn_probs.min().item():.6e}/{attn_probs.max().item():.6e}/{attn_probs.mean().item():.6e}")
        
        # Check for numerical issues
        print(f"Contains zeros: {(attn_probs == 0).any().item()}")
        print(f"Contains NaN: {torch.isnan(attn_probs).any().item()}")
        print(f"Contains Inf: {torch.isinf(attn_probs).any().item()}")
        
        # Check distribution validity
        row_sums = attn_probs.sum(dim=-1)
        print(f"Row sums min/max/mean: {row_sums.min().item():.6f}/{row_sums.max().item():.6f}/{row_sums.mean().item():.6f}")
        print(f"Rows sum to ~1: {torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-2)}")
    
    # Apply numerical safeguards
    # 1. Ensure positive values
    attn_probs = attn_probs.clamp(min=eps)
    
    # 2. Normalize to ensure it sums to 1.0 along attention dimension
    attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)
    
    if debug:
        print("\\nAfter preprocessing:")
        print(f"Min/max/mean: {attn_probs.min().item():.6e}/{attn_probs.max().item():.6e}/{attn_probs.mean().item():.6e}")
        row_sums = attn_probs.sum(dim=-1)
        print(f"Row sums min/max/mean: {row_sums.min().item():.6f}/{row_sums.max().item():.6f}/{row_sums.mean().item():.6f}")
    
    # Compute entropy: -sum(p * log(p))
    log_probs = torch.log(attn_probs)
    entropy = -torch.sum(attn_probs * log_probs, dim=-1)
    
    if debug:
        print("\\nEntropy results:")
        print(f"Entropy shape: {entropy.shape}")
        print(f"Entropy min/max/mean: {entropy.min().item():.4f}/{entropy.max().item():.4f}/{entropy.mean().item():.4f}")
        
        # Compute theoretical maximum entropy (uniform distribution)
        seq_len = attn_probs.size(-1)
        max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float))
        print(f"Theoretical max entropy (log(seq_len)): {max_entropy.item():.4f}")
        
        # Check if entropy is at maximum (uniform attention)
        print(f"Percentage of maximum entropy: {entropy.mean().item()/max_entropy.item()*100:.2f}%")
    
    return entropy

# Get the raw attention patterns from the model for analysis
try:
    # Get a batch of data
    inputs = next(iter(validation_dataloader))
    if isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    else:
        inputs = {"input_ids": inputs[0].to(device)}
    
    # Run model with attention outputs
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Extract attention patterns
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        attn_list = outputs.attentions
        if len(attn_list) > 0:
            # Create a detailed visualization of attention patterns and entropy
            num_layers = len(attn_list)
            fig, axes = plt.subplots(num_layers, 2, figsize=(12, num_layers*3))
            
            layer_entropies = []
            layer_entropies_norm = []
            
            for layer_idx in range(num_layers):
                attn = attn_list[layer_idx]
                
                # Compute entropy for this layer's attention
                print(f"\\n=== Analyzing Layer {layer_idx} Attention ====")
                layer_entropy = compute_improved_entropy(attn, debug=True)
                
                # Save mean entropy per head
                head_entropies = layer_entropy.mean(dim=(0, 1))  # Average over batch and sequence
                layer_entropies.append(head_entropies)
                
                # Normalize by max possible entropy
                seq_len = attn.size(-1)
                max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float, device=attn.device))
                norm_entropies = head_entropies / max_entropy.item()
                layer_entropies_norm.append(norm_entropies)
                
                # Plot attention pattern for first head
                if isinstance(axes, np.ndarray) and len(axes.shape) > 1:  # multiple rows and cols
                    ax1 = axes[layer_idx, 0]
                    ax2 = axes[layer_idx, 1]
                else:  # only 1 layer, so axes is 1D
                    ax1 = axes[0]
                    ax2 = axes[1]
                
                # Plot attention pattern
                attn_pattern = attn[0, 0].cpu().numpy()  # First batch, first head
                im = ax1.imshow(attn_pattern, cmap='viridis')
                ax1.set_title(f'Layer {layer_idx} - Head 0 Attention')
                ax1.set_xlabel('Position (To)')
                ax1.set_ylabel('Position (From)')
                plt.colorbar(im, ax=ax1)
                # Set proper limits for attention values (0 to 1)
                im.set_clim(0, 1.0)
                
                # Plot entropy values for all heads
                ax2.bar(range(len(head_entropies)), head_entropies.cpu().numpy())
                ax2.axhline(y=max_entropy.item(), color='r', linestyle='--', alpha=0.7, label='Max Entropy')
                ax2.set_title(f'Layer {layer_idx} - Head Entropies')
                ax2.set_xlabel('Head Index')
                ax2.set_ylabel('Entropy')
                ax2.legend()
                
                # Add entropy values as text on the bars
                for i, v in enumerate(head_entropies):
                    ax2.text(i, v.item() + 0.1, f'{v.item():.2f}', ha='center')
            
            plt.tight_layout()
            plt.show()
            
            # Create a heatmap of entropy across all layers and heads
            if num_layers > 1:
                all_entropies = torch.stack(layer_entropies).cpu().numpy()
                plt.figure(figsize=(10, 6))
                plt.imshow(all_entropies, cmap='viridis', aspect='auto')
                plt.clim(0, max(0.1, all_entropies.max()))  # Ensure non-zero range
                plt.colorbar(label='Entropy')
                plt.title('Entropy Heatmap Across All Layers and Heads')
                plt.xlabel('Head Index')
                plt.ylabel('Layer Index')
                
                # Add text annotations for each cell
                for i in range(all_entropies.shape[0]):
                    for j in range(all_entropies.shape[1]):
                        text = plt.text(j, i, f'{all_entropies[i, j]:.2f}',
                                      ha="center", va="center", color="w")
                
                plt.tight_layout()
                plt.show()
                
                # Plot normalized entropy (as percentage of maximum)
                all_norm_entropies = torch.stack(layer_entropies_norm).cpu().numpy() * 100  # as percentage
                plt.figure(figsize=(10, 6))
                plt.imshow(all_norm_entropies, cmap='viridis', aspect='auto', vmin=0, vmax=100)
                plt.colorbar(label='% of Max Entropy')
                plt.title('Normalized Entropy (% of Maximum)')
                plt.xlabel('Head Index')
                plt.ylabel('Layer Index')
                
                # Add text annotations for each cell
                for i in range(all_norm_entropies.shape[0]):
                    for j in range(all_norm_entropies.shape[1]):
                        text = plt.text(j, i, f'{all_norm_entropies[i, j]:.1f}%',
                                      ha="center", va="center", color="w")
                
                plt.tight_layout()
                plt.show()
        else:
            print("No attention tensors returned by the model")
    else:
        print("Model outputs don't include attention weights")
except Exception as e:
    print(f"Error in entropy analysis: {e}")"""
        
        # Convert Cell 18
        cell_18_lines = []
        for line in cell_18_source.split('\n'):
            cell_18_lines.append(line + '\n')
        if cell_18_lines and cell_18_lines[-1].endswith('\n'):
            cell_18_lines[-1] = cell_18_lines[-1].rstrip('\n')
            
        # Fix cell 20 - Fix entropy visualization in comparison plots
        cell_20_source = """# Create a visual comparing entropy and gradient distributions
plt.figure(figsize=(10, 6))

# Function to properly calculate entropy
def calculate_proper_entropy(attn_tensor, eps=1e-8):
    # Calculate entropy with proper normalization and numerical stability
    # Get attention shape
    batch_size, num_heads, seq_len, _ = attn_tensor.shape
    
    # Reshape for processing
    attn_flat = attn_tensor.view(batch_size * num_heads * seq_len, -1)
    
    # Handle numerical issues - ensure positive values and proper normalization
    attn_flat = attn_flat.clamp(min=eps)
    attn_flat = attn_flat / attn_flat.sum(dim=-1, keepdim=True)
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(attn_flat * torch.log(attn_flat), dim=-1)
    
    # Reshape back to per-head format and average
    entropy = entropy.view(batch_size, num_heads, seq_len)
    entropy = entropy.mean(dim=(0, 2))  # Average over batch and sequence
    
    # Normalize by maximum possible entropy (log of sequence length)
    max_entropy = torch.log(torch.tensor(attn_tensor.size(-1), dtype=torch.float, device=attn_tensor.device))
    
    # View as layers x heads
    return entropy.view(-1, num_heads)

# Get attention outputs to calculate entropy directly
try:
    # Get sample data
    inputs = next(iter(validation_dataloader))
    if isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    else:
        inputs = {"input_ids": inputs[0].to(device)}
    
    # Forward pass with attention outputs
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Calculate entropy from raw attention
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        # Extract attention tensors
        attentions = outputs.attentions
        
        # Print diagnostic info
        print(f"Number of attention layers: {len(attentions)}")
        first_attn = attentions[0]
        print(f"Attention shape: {first_attn.shape}")
        print(f"Attention statistics - min: {first_attn.min().item():.6f}, max: {first_attn.max().item():.6f}")
        
        # Check if attention sums to 1 along correct dimension
        attn_sum = first_attn.sum(dim=-1)
        print(f"Attention sum along last dim - min: {attn_sum.min().item():.6f}, max: {attn_sum.max().item():.6f}")
        
        # Calculate proper entropy for all layers
        all_entropies = torch.cat([calculate_proper_entropy(attn) for attn in attentions])
        
        # Print entropy statistics
        print(f"Calculated entropy shape: {all_entropies.shape}")
        print(f"Entropy statistics - min: {all_entropies.min().item():.6f}, max: {all_entropies.max().item():.6f}")
        
        # Side by side plots
        plt.subplot(1, 2, 1)
        im1 = plt.imshow(all_entropies.cpu().numpy(), cmap='viridis', aspect='auto')
        plt.clim(0, max(0.1, all_entropies.max().item()))  # Ensure proper visualization range
        plt.colorbar(im1, label='Entropy')
        plt.title(f'Properly Calculated Attention Entropy (max={all_entropies.max().item():.4f})')
        plt.xlabel('Head Index')
        plt.ylabel('Layer Index')
        
        # Use the gradient tensor from debug metrics
        plt.subplot(1, 2, 2)
        im2 = plt.imshow(debug_grads.detach().cpu().numpy(), cmap='plasma', aspect='auto')
        plt.colorbar(im2, label='Gradient Norm')
        plt.title('Gradient Norms')
        plt.xlabel('Head Index')
        plt.ylabel('Layer Index')
        
        plt.tight_layout()
        plt.show()
        
        # Create a scatter plot to show relationship
        plt.figure(figsize=(8, 6))
        entropy_flat = all_entropies.flatten().cpu().numpy()
        grad_flat = debug_grads.flatten().cpu().numpy()
        
        plt.scatter(entropy_flat, grad_flat, alpha=0.7)
        plt.xlabel('Entropy (higher = less focused)')
        plt.ylabel('Gradient Norm (higher = more impact)')
        plt.title('Entropy vs Gradient Relationship')
        plt.grid(alpha=0.3)
        plt.show()
        
    else:
        print("Model did not return attention tensors")
except Exception as e:
    print(f"Error in entropy calculation: {e}")
    
    # Fallback - use debug_entropy that was already collected
    # Plot entropy with a manual scale to force visibility
    plt.subplot(1, 2, 1)
    
    # Enforce a minimum scale for visibility
    entropy_data = debug_entropy.detach().cpu().numpy()
    im1 = plt.imshow(entropy_data, cmap='viridis', aspect='auto', vmin=0, vmax=max(0.1, entropy_data.max()))
    plt.colorbar(im1, label='Entropy')
    plt.title(f'Attention Entropy Values (max={entropy_data.max():.4f})')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')

    # Gradient subplot
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(debug_grads.detach().cpu().numpy(), cmap='plasma', aspect='auto')
    plt.colorbar(im2, label='Gradient Norm')
    plt.title('Gradient Norms')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    
    plt.tight_layout()
    plt.show()"""
        
        # Convert Cell 20
        cell_20_lines = []
        for line in cell_20_source.split('\n'):
            cell_20_lines.append(line + '\n')
        if cell_20_lines and cell_20_lines[-1].endswith('\n'):
            cell_20_lines[-1] = cell_20_lines[-1].rstrip('\n')
            
        # Update the cells in the notebook
        notebook['cells'][17]['source'] = cell_17_lines
        notebook['cells'][18]['source'] = cell_18_lines
        notebook['cells'][20]['source'] = cell_20_lines

        # Update version to v0.0.34 in first cell if not already updated
        if notebook['cells'][0]['cell_type'] == 'markdown':
            cell_0_source = ''.join(notebook['cells'][0]['source']) if isinstance(notebook['cells'][0]['source'], list) else notebook['cells'][0]['source']
            
            if "v0.0.34" not in cell_0_source:
                # Update version number and add changelog entry
                cell_0_source = cell_0_source.replace("v0.0.33", "v0.0.34")
                
                # Add entry for v0.0.34 if not already there
                if "### New in v0.0.34:" not in cell_0_source:
                    # Find position after "This allows models to form more efficient neural structures during training."
                    split_text = "This allows models to form more efficient neural structures during training."
                    parts = cell_0_source.split(split_text)
                    if len(parts) > 1:
                        new_entry = """

### New in v0.0.34:
- Fixed attention visualization scaling
- Improved colormap limits for better pattern visibility
- Added proper colorbar labels
"""
                        cell_0_source = parts[0] + split_text + new_entry + parts[1].lstrip()
                        
                # Convert to list if needed
                if isinstance(notebook['cells'][0]['source'], list):
                    cell_0_lines = []
                    for line in cell_0_source.split('\n'):
                        cell_0_lines.append(line + '\n')
                    if cell_0_lines and cell_0_lines[-1].endswith('\n'):
                        cell_0_lines[-1] = cell_0_lines[-1].rstrip('\n')
                    notebook['cells'][0]['source'] = cell_0_lines
                else:
                    notebook['cells'][0]['source'] = cell_0_source
                
                print("Updated version to v0.0.34 and added changelog entry")
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Successfully applied attention visualization fixes to cells 17, 18, and 20")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_attention_visualization(notebook_path)