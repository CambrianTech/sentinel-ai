#!/usr/bin/env python
# Fix entropy calculation and visualization to address zero entropy issue

import json
from pathlib import Path

def fix_entropy_calculation_direct(notebook_path):
    """Fix entropy calculation and visualization to ensure non-zero values."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find the cell that creates the entropy-gradient comparison
        entropy_viz_cell_idx = None
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'Create a visual comparing entropy and gradient distributions' in source:
                    entropy_viz_cell_idx = i
                    print(f"Found entropy visualization cell at index {i}")
                    break
        
        if entropy_viz_cell_idx is None:
            print("ERROR: Could not find entropy visualization cell")
            return False
        
        # Replace with fixed visualization code that calculates entropy correctly
        fixed_content = """# Create a visual comparing entropy and gradient distributions
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
    im1 = plt.imshow(entropy_data, cmap='viridis', aspect='auto', vmin=0, vmax=0.1)
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
        
        # Update the cell content
        cell = notebook['cells'][entropy_viz_cell_idx]
        cell['source'] = fixed_content.split('\n')
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Fixed entropy calculation and visualization")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_entropy_calculation_direct(notebook_path)