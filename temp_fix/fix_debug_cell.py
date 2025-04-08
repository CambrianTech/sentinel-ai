#!/usr/bin/env python
# Fix debug cell to avoid execution issues

import json
from pathlib import Path

def fix_debug_cell(notebook_path):
    """Fix specific issues in the debug cell."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find the debug cell
        debug_cell_idx = None
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'debug_entropy, debug_grads = controller.collect_head_metrics(' in source:
                    debug_cell_idx = i
                    print(f"Found debug cell at index {i}")
                    break
        
        if debug_cell_idx is None:
            print("ERROR: Could not find debug cell")
            return False
        
        # Get the cell content
        cell = notebook['cells'][debug_cell_idx]
        
        # Create a completely new, fixed debug cell
        fixed_content = """# Debug: Let's check the actual entropy values we're dealing with
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
                    \"\"\"Compute entropy with better numerical stability.\"\"\"
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
                plt.imshow(attn[0, head_idx].cpu().numpy(), cmap='viridis')
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
        
        # Replace the cell content
        cell['source'] = fixed_content.split('\n')
        
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"Fixed debug cell with proper error handling")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_debug_cell(notebook_path)