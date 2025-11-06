"""
Example code showing how to use the dual-mode pruning in a Colab notebook.
This can be added to PruningAndFineTuningColab.py.
"""

# [START: Pruning Mode Selection Cell]

# Add imports for pruning modes
try:
    from sentinel.pruning.dual_mode_pruning import prune_head_in_model, apply_pruning_hooks, PruningMode, get_model_info
except ImportError:
    # For Colab compatibility
    !pip install --quiet git+https://github.com/CambrianTech/sentinel-ai.git
    from sentinel.pruning.dual_mode_pruning import prune_head_in_model, apply_pruning_hooks, PruningMode, get_model_info

# Add widget for pruning mode selection
try:
    from ipywidgets import widgets
    from IPython.display import display
    
    pruning_mode = PruningMode.ADAPTIVE  # Default
    
    # Create mode selector widget
    mode_selector = widgets.RadioButtons(
        options=[
            ('Adaptive (quality focus)', PruningMode.ADAPTIVE), 
            ('Compressed (size/speed focus)', PruningMode.COMPRESSED)
        ],
        description='Pruning Mode:',
        disabled=False
    )
    
    # Description text
    adaptive_desc = widgets.HTML(
        value="""<div style="margin-left: 20px; margin-bottom: 10px; color: #555;">
            <b>Adaptive</b>: Temporarily zeros weights, allows recovery during fine-tuning.<br>
            <small>• Better for maximizing quality</small><br>
            <small>• Allows heads to recover if needed</small><br>
            <small>• Does not reduce model size</small>
        </div>"""
    )
    
    compressed_desc = widgets.HTML(
        value="""<div style="margin-left: 20px; margin-bottom: 20px; color: #555;">
            <b>Compressed</b>: Permanently zeros weights, prevents recovery.<br>
            <small>• Better for deployment/efficiency</small><br>
            <small>• Maintains true sparsity during training</small><br>
            <small>• Can be exported as smaller model</small>
        </div>"""
    )
    
    # Update pruning mode when selection changes
    def on_mode_change(change):
        global pruning_mode
        if change['type'] == 'change' and change['name'] == 'value':
            pruning_mode = change['new']
            print(f"Pruning mode set to: {pruning_mode}")
            if pruning_mode == PruningMode.COMPRESSED:
                print("  • Heads will be permanently pruned")
                print("  • No recovery during fine-tuning")
                print("  • Better for deployment/efficiency")
            else:
                print("  • Pruned heads may recover during fine-tuning")
                print("  • Better for maximizing model quality")
    
    mode_selector.observe(on_mode_change, names='value')
    
    # Display widget with descriptions
    display(widgets.HTML("<h3>Pruning Mode Selection</h3>"))
    display(mode_selector)
    display(adaptive_desc)
    display(compressed_desc)
    
except ImportError:
    print("Widget support not available. Using default pruning mode: ADAPTIVE")

# [END: Pruning Mode Selection Cell]

# [START: Enhanced Pruning Cell]

def apply_pruning(model, strategy, pruning_level, prompt, verbose=True):
    """
    Apply pruning with the selected mode.
    
    Args:
        model: The transformer model
        strategy: Pruning strategy ("random", "entropy", etc.)
        pruning_level: Fraction of heads to prune (0.0-1.0)
        prompt: Text prompt for evaluation
        verbose: Whether to print verbose output
        
    Returns:
        List of pruned head indices
    """
    if verbose:
        print(f"Applying {strategy} pruning with level {pruning_level} in {pruning_mode} mode")
    
    # Collect model information before pruning
    if verbose:
        before_info = get_model_info(model)
        print(f"Model before pruning: {before_info['size_mb']:.2f} MB, " 
              f"{before_info['total_params']:,} parameters")
    
    # Calculate head importance based on strategy
    # (existing code to calculate importance)
    # ...
    
    # Get head indices to prune
    # (existing code to select heads)
    # ...
    
    # Apply pruning using the selected mode
    pruned_heads = []
    for layer_idx, head_idx in heads_to_prune:
        if prune_head_in_model(model, layer_idx, head_idx, mode=pruning_mode, verbose=verbose):
            pruned_heads.append((layer_idx, head_idx))
    
    # If using compressed mode, add hooks to maintain pruned state
    hooks = []
    if pruning_mode == PruningMode.COMPRESSED:
        hooks = apply_pruning_hooks(model, pruned_heads, mode=pruning_mode, verbose=verbose)
    
    # Collect model information after pruning
    if verbose:
        after_info = get_model_info(model)
        print(f"Model after pruning: {after_info['size_mb']:.2f} MB, "
              f"{after_info['nonzero_params']:,} non-zero parameters")
        print(f"Sparsity: {after_info['sparsity']:.2%}")
    
    return pruned_heads, hooks

# [END: Enhanced Pruning Cell]

# [START: Fine-tuning with Mode-Specific Behavior]

def fine_tune_pruned_model(model, pruned_heads, dataset, epochs, batch_size=4, learning_rate=5e-5):
    """
    Fine-tune the pruned model with mode-specific behavior.
    
    Args:
        model: The pruned model
        pruned_heads: List of pruned head indices
        dataset: Dataset to use for fine-tuning
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        
    Returns:
        Fine-tuned model and training metrics
    """
    print(f"Fine-tuning pruned model in {pruning_mode} mode")
    
    # For compressed mode, we need to ensure gradients don't flow to pruned heads
    if pruning_mode == PruningMode.COMPRESSED:
        print("Using compressed mode fine-tuning (preventing recovery)")
        # Hooks applied during pruning will ensure pruned heads stay pruned
    else:
        print("Using adaptive mode fine-tuning (allowing recovery)")
        # No special handling needed - weights can recover naturally
    
    # Rest of fine-tuning code remains the same
    # ...
    
    # After fine-tuning, analyze recovery if in adaptive mode
    if pruning_mode == PruningMode.ADAPTIVE:
        # Check how many pruned heads recovered
        recovered_heads = []
        for layer_idx, head_idx in pruned_heads:
            # (code to check if this head has non-zero weights now)
            # ...
            if head_recovered:
                recovered_heads.append((layer_idx, head_idx))
        
        print(f"{len(recovered_heads)}/{len(pruned_heads)} pruned heads recovered during fine-tuning")
    
    return model, metrics

# [END: Fine-tuning with Mode-Specific Behavior]