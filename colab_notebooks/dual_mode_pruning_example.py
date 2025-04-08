"""
Example Script for Using Dual-Mode Pruning in Colab

This example script demonstrates how to use the dual-mode pruning system
with both adaptive and compressed modes.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentinel.pruning.dual_mode_pruning import prune_head_in_model, apply_pruning_hooks, PruningMode, get_model_info

def run_pruning_example(model_name="distilgpt2", pruning_mode="adaptive"):
    """
    Demonstrate pruning a model with the specified mode.
    
    Args:
        model_name: Name of the HuggingFace model to use
        pruning_mode: Either "adaptive" or "compressed"
    
    Returns:
        Tuple of (model, hooks)
    """
    # Set pruning mode
    mode = PruningMode.ADAPTIVE if pruning_mode.lower() == "adaptive" else PruningMode.COMPRESSED
    
    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get initial model info
    initial_info = get_model_info(model)
    print(f"Initial model size: {initial_info['size_mb']:.2f} MB")
    print(f"Initial sparsity: {initial_info['sparsity']:.2%}")
    
    # Prune selected heads
    print(f"\nPruning heads using {mode.value.upper()} mode...")
    heads_to_prune = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for layer_idx, head_idx in heads_to_prune:
        prune_head_in_model(model, layer_idx, head_idx, mode=mode)
    
    # Get post-pruning model info
    pruned_info = get_model_info(model)
    print(f"Post-pruning sparsity: {pruned_info['sparsity']:.2%}")
    print(f"Zeroed parameters: {pruned_info['zero_params'] - initial_info['zero_params']:,}")
    
    # Apply pruning hooks
    print("\nApplying hooks to maintain pruned state during training...")
    hooks = apply_pruning_hooks(model, heads_to_prune, mode=mode)
    
    # Demonstrate usage with training
    print("\nSimulating training with pruned model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # For adaptive mode, patch the optimizer if first layer has a hook
    if mode == PruningMode.ADAPTIVE:
        attn_module = model.transformer.h[0].attn
        optimizer_hook = getattr(attn_module, '_optimizer_hook', None)
        if optimizer_hook:
            print("Patching optimizer with re-zeroing hook")
            optimizer.orig_step = optimizer.step
            optimizer.step = lambda *a, **kw: optimizer_hook(optimizer, {'args': a, 'kwargs': kw})
    
    # Test generation before training
    print("\nGeneration BEFORE training:")
    input_text = "The meaning of life is"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(input_ids, max_length=30, do_sample=True, temperature=0.7)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    
    # Simulate training
    for step in range(3):
        inputs = tokenizer("This is a training example to test pruning behavior.", return_tensors="pt")
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step+1}, loss: {loss.item():.4f}")
    
    # Test generation after training
    print("\nGeneration AFTER training:")
    with torch.no_grad():
        output = model.generate(input_ids, max_length=30, do_sample=True, temperature=0.7)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    
    # For experiments only: remove hooks
    if mode == PruningMode.ADAPTIVE:
        print("\nFor experimentation: Removing hooks and optimizer patching to allow heads to recover")
        for hook in hooks:
            hook.remove()
        
        # Restore original optimizer behavior
        if hasattr(optimizer, 'orig_step'):
            optimizer.step = optimizer.orig_step
            delattr(optimizer, 'orig_step')
    
    return model, hooks

# Example usage
if __name__ == "__main__":
    # Adaptive mode example
    model_adaptive, hooks_adaptive = run_pruning_example(pruning_mode="adaptive")
    
    print("\n" + "="*40 + "\n")
    
    # Compressed mode example
    model_compressed, hooks_compressed = run_pruning_example(pruning_mode="compressed")