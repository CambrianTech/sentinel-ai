# Evaluate after pruning
print("\nEvaluating pruned model performance...")
pruned_loss, pruned_ppl = evaluate_model(model, val_dataloader)
print(f"After pruning: loss: {pruned_loss:.4f}, perplexity: {pruned_ppl:.2f}")

# Generate example text with pruned model
pruned_generation = generate_text(
    model, tokenizer, 
    prompt="Artificial intelligence is becoming increasingly important because"
)
print("\nAfter pruning generation example:")
print(pruned_generation)

# Record metrics after pruning
progress.update(step=1, loss=pruned_loss, perplexity=pruned_ppl, 
               gate_values=get_gate_values(model),
               generation_sample=pruned_generation)

# Save gate visualization
gate_viz_path = os.path.join(run_dir, "gate_values.png")
visualize_gate_values(get_gate_values(model), gate_viz_path)

# Fine-tune pruned model
print("\nFine-tuning pruned model...")

try:
    # Clear GPU memory before fine-tuning
    clear_gpu_memory()
    
    # First check if we should use CPU directly
    # If we're already on CPU or memory is tight, just use CPU
    import psutil
    ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    if DEVICE == "cpu" or (DEVICE == "cuda" and ram_gb < 2.0):
        print("Using CPU for fine-tuning due to memory constraints")
        model = model.cpu()
        DEVICE = "cpu"
        
        # Update loaders for CPU
        train_dataloader = torch.utils.data.DataLoader(
            train_dataloader.dataset, 
            batch_size=max(1, train_dataloader.batch_size // 2),
            shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataloader.dataset, 
            batch_size=max(1, val_dataloader.batch_size // 2),
            shuffle=False
        )
        
        # Reduce epochs on CPU
        num_epochs = min(num_epochs, 1)
    
    # Use the context manager for better memory efficiency
    with autocast_if_available():
        fine_tune_results = fine_tune_model(
            model, 
            train_dataloader, 
            val_dataloader, 
            tokenizer,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            progress_tracker=progress
        )
        
except RuntimeError as e:
    if "CUDA" in str(e):
        print(f"\nCUDA error during fine-tuning: {e}")
        print("\nAttempting to continue with CPU...\n")
        
        # Try moving to CPU before any other operations
        try:
            # Create a fresh model instance on CPU to avoid CUDA error propagation
            if "cpu_model" not in locals():
                clear_gpu_memory()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                print("Loading model from scratch on CPU...")
                cpu_model, cpu_tokenizer = load_model_and_tokenizer(model_name, cache_dir=MODEL_CACHE_DIR)
                cpu_model = cpu_model.cpu()
                
                # Apply same pruning to the new model
                head_importances = get_head_importances(cpu_model, val_dataloader, strategy=strategy)
                pruned_heads = prune_heads(cpu_model, head_importances, pruning_level=pruning_level)
                
                # Update model reference and move to CPU
                model = cpu_model
                tokenizer = cpu_tokenizer
                DEVICE = "cpu"
                
                # Update loaders for CPU
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataloader.dataset, 
                    batch_size=2,  # Small batch size for CPU
                    shuffle=True
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataloader.dataset, 
                    batch_size=2,  # Small batch size for CPU
                    shuffle=False
                )
            
            # Try with reduced parameters
            fine_tune_results = fine_tune_model(
                model, 
                train_dataloader, 
                val_dataloader, 
                tokenizer,
                learning_rate=learning_rate / 2,  # Lower learning rate
                num_epochs=1,  # Reduce to single epoch
                progress_tracker=progress
            )
            
        except Exception as cpu_error:
            print(f"CPU fallback also failed: {cpu_error}")
            # Provide minimal output to continue using values we already calculated
            fine_tune_results = {
                "final_loss": pruned_loss,  # Use pruned model metrics that we calculated earlier
                "final_perplexity": pruned_ppl,
                "steps": 0
            }
    else:
        print(f"Error during fine-tuning: {e}")
        # Provide minimal output to continue using values we already calculated
        fine_tune_results = {
            "final_loss": pruned_loss,  # Use pruned model metrics that we calculated earlier
            "final_perplexity": pruned_ppl,
            "steps": 0
        }