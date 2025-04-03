"""
Fine-tuning implementation for pruned models with improved stability for large models
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from functools import partial
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import FlaxAutoModelForCausalLM
import logging
import math
import time
import gc

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ImprovedFineTuner:
    """Fine-tunes a pruned model to recover performance with improved stability"""
    
    def __init__(self, pruning_module, dataset_name="openwebtext", dataset_config=None, batch_size=4):
        self.pruning_module = pruning_module
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.max_seq_length = 128  # Modest sequence length for faster training
        self.train_state = None
        self.metrics_history = []
        self.use_synthetic_data = False  # New flag for forcing synthetic data
        self.use_rng_keys_for_dropout = True  # Enable by default to fix dropout RNG issue
        
        # Detect number of devices
        self.devices = jax.devices()
        self.n_devices = len(self.devices)
        
        # Detect model family and adjust parameters accordingly
        self.is_opt_model = 'opt' in self.pruning_module.model_name.lower()
        
        if self.n_devices > 1:
            logger.info(f"Using {self.n_devices} devices for training")
            self.batch_size = max(self.batch_size, self.n_devices)
            # Make batch size divisible by device count
            self.batch_size = (self.batch_size // self.n_devices) * self.n_devices
            logger.info(f"Adjusted batch size to {self.batch_size} for multi-device training")
            
        # Adjust batch size for large models
        self._adjust_batch_size_for_model_size()
    
    def _adjust_batch_size_for_model_size(self):
        """Adjust batch size based on model size to prevent OOM errors"""
        model_name = self.pruning_module.model_name.lower()
        
        # Detect large models
        if "1.3b" in model_name or "large" in model_name:
            # Reduce batch size for large models
            old_batch_size = self.batch_size
            self.batch_size = max(1, self.batch_size // 4)
            logger.info(f"Reduced batch size from {old_batch_size} to {self.batch_size} for large model {model_name}")
        elif "medium" in model_name or "base" in model_name or "350m" in model_name:
            # Reduce batch size for medium models
            old_batch_size = self.batch_size
            self.batch_size = max(1, self.batch_size // 2)
            logger.info(f"Reduced batch size from {old_batch_size} to {self.batch_size} for medium-sized model {model_name}")
    
    def _prepare_dataset(self):
        """Load and prepare the dataset for fine-tuning"""
        # If synthetic data is requested, skip real dataset loading
        if self.use_synthetic_data:
            logger.info("Using synthetic data as requested")
            return self._prepare_synthetic_dataset()
        
        try:
            # Try to load a small portion of the dataset for faster loading
            if self.dataset_config:
                logger.info(f"Loading dataset {self.dataset_name} with config {self.dataset_config}")
                dataset = load_dataset(self.dataset_name, self.dataset_config, split="train[:5000]")
            else:
                logger.info(f"Loading dataset {self.dataset_name}")
                dataset = load_dataset(self.dataset_name, split="train[:5000]")
                
            logger.info(f"Dataset loaded: {len(dataset)} examples")
            
            # Process dataset
            tokenizer = self.pruning_module.tokenizer
            
            # Ensure tokenizer has pad_token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.eos_token = "[PAD]"
                logger.info(f"Set pad_token to {tokenizer.pad_token}")
            
            def tokenize_function(examples):
                # Tokenize the texts
                if "text" in examples:
                    texts = examples["text"]
                else:
                    # Try to find text field (wikitext has different format)
                    keys = examples.keys()
                    text_key = next((k for k in keys if "text" in k.lower()), None)
                    if text_key:
                        texts = examples[text_key]
                    else:
                        # If no text field found, concatenate all string fields
                        texts = []
                        for i in range(len(examples[next(iter(keys))])):
                            example_text = " ".join(str(examples[k][i]) for k in keys 
                                                if isinstance(examples[k][i], str))
                            texts.append(example_text)
                
                # Add truncation to prevent very long sequences
                tokenized = tokenizer(
                    texts, 
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding="max_length",
                    return_tensors="np"
                )
                return tokenized
            
            # Remove columns that aren't strings
            columns_to_remove = []
            for col in dataset.column_names:
                if isinstance(dataset[0][col], (int, float, bool)) or dataset[0][col] is None:
                    continue
                columns_to_remove.append(col)
            
            try:
                tokenized_dataset = dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=1,
                    remove_columns=columns_to_remove
                )
                
                # Convert to list of batches for simpler processing
                dataloader = []
                for i in range(0, len(tokenized_dataset), self.batch_size):
                    end = min(i + self.batch_size, len(tokenized_dataset))
                    batch_samples = tokenized_dataset[i:end]
                    
                    batch = {
                        "input_ids": np.array([sample["input_ids"] for sample in batch_samples]),
                        "attention_mask": np.array([sample["attention_mask"] for sample in batch_samples]),
                    }
                    
                    # Add labels
                    batch["labels"] = batch["input_ids"].copy()
                    
                    dataloader.append(batch)
                
                logger.info(f"Created {len(dataloader)} batches")
                return dataloader
                
            except Exception as e:
                logger.error(f"Error creating batches: {e}")
                logger.info("Falling back to synthetic data")
                return self._prepare_synthetic_dataset()
                
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            logger.info("Falling back to synthetic data for training")
            import traceback
            traceback.print_exc()
            return self._prepare_synthetic_dataset()
    
    def _prepare_synthetic_dataset(self):
        """Create synthetic data for training when dataset loading fails"""
        tokenizer = self.pruning_module.tokenizer
        
        # Ensure tokenizer has pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token = "[PAD]"
            logger.info(f"Set pad_token to {tokenizer.pad_token}")
        
        # Generate random token IDs (avoid special tokens)
        vocab_size = tokenizer.vocab_size
        
        # Get special token IDs (safely)
        special_tokens = set()
        for token_name in ['pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id']:
            token_id = getattr(tokenizer, token_name, None)
            if token_id is not None:
                special_tokens.add(token_id)
        
        logger.info(f"Creating synthetic dataset with vocab_size={vocab_size}, special_tokens={special_tokens}")
        
        # Create 100 samples of random token sequences
        samples = []
        for _ in range(100):
            # Generate random length between 10 and max_seq_length
            length = np.random.randint(10, self.max_seq_length)
            
            # Generate random token IDs
            token_ids = np.random.randint(0, min(30000, vocab_size), size=length)
            
            # Replace special tokens with normal tokens
            for i, token_id in enumerate(token_ids):
                if token_id in special_tokens:
                    token_ids[i] = (token_id + 1) % min(30000, vocab_size)
                    # Make sure we're not just cycling through special tokens
                    while token_ids[i] in special_tokens:
                        token_ids[i] = (token_ids[i] + 1) % min(30000, vocab_size)
            
            # Create sample
            sample = {
                "input_ids": token_ids,
                "attention_mask": np.ones_like(token_ids),
                "labels": token_ids.copy()
            }
            samples.append(sample)
        
        # Create batches
        batches = []
        for i in range(0, len(samples), self.batch_size):
            batch_samples = samples[i:i+self.batch_size]
            
            # Pad to the same length within batch
            max_len = max(len(s["input_ids"]) for s in batch_samples)
            
            batch = {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }
            
            for sample in batch_samples:
                pad_len = max_len - len(sample["input_ids"])
                batch["input_ids"].append(np.pad(sample["input_ids"], (0, pad_len), 
                                              constant_values=tokenizer.pad_token_id))
                batch["attention_mask"].append(np.pad(sample["attention_mask"], (0, pad_len), 
                                                  constant_values=0))
                batch["labels"].append(np.pad(sample["labels"], (0, pad_len), 
                                          constant_values=tokenizer.pad_token_id))
            
            # Convert to arrays
            batch = {k: np.array(v) for k, v in batch.items()}
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} synthetic batches")
        return batches
    
    def _create_train_state(self, params, learning_rate=5e-5):
        """Create a training state for the fine-tuning process"""
        # Create optimizer
        # Use AdamW instead of Adam for better weight decay handling
        optimizer = optax.adamw(learning_rate, weight_decay=0.01)
        
        # Create train state
        model = self.pruning_module.model
        return train_state.TrainState.create(
            apply_fn=model.__call__,
            params=params,
            tx=optimizer
        )
    
    def _loss_fn(self, params, batch):
        """Loss function for the language modeling task"""
        model = self.pruning_module.model
        
        # Extract labels from batch but don't pass them to the model
        labels = batch.pop("labels", None)
        
        # Extract dropout_rng if present for handling dropout
        dropout_rng = batch.pop("dropout_rng", None)
        
        # Check if we need to handle NaN or Inf in input
        for k, v in batch.items():
            if jnp.isnan(v).any() or jnp.isinf(v).any():
                # Replace NaN and Inf with zeros
                batch[k] = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                logger.warning(f"Found NaN or Inf in {k}, replaced with zeros")
        
        try:
            # Prepare the model call arguments
            model_kwargs = {"params": params}
            
            # Add dropout_rng if present
            if dropout_rng is not None:
                model_kwargs["dropout_rng"] = dropout_rng
                
            # Add train=True for models that need it (except OPT)
            if not self.is_opt_model:
                model_kwargs["train"] = True
            
            # Get logits from model with appropriate arguments
            outputs = model(**batch, **model_kwargs)
            logits = outputs.logits
            
            # Add labels back to batch for next iteration
            batch["labels"] = labels
            
            # Create loss mask (don't compute loss for padding tokens)
            loss_mask = (labels != self.pruning_module.tokenizer.pad_token_id)
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            shift_mask = loss_mask[:, 1:]
            
            # Calculate cross entropy loss
            loss = optax.softmax_cross_entropy_with_integer_labels(
                shift_logits, shift_labels
            )
            
            # Check for NaN and Inf in loss
            if jnp.isnan(loss).any() or jnp.isinf(loss).any():
                logger.warning(f"Found NaN or Inf in loss: {loss}")
                # Use a large but finite value instead of NaN or Inf
                loss = jnp.nan_to_num(loss, nan=1e3, posinf=1e3, neginf=1e3)
            
            # Apply mask and calculate mean - safely with jnp.where to avoid division by zero
            masked_loss = loss * shift_mask
            mask_sum = shift_mask.sum()
            
            # Safe division with fallback
            loss = jnp.where(
                mask_sum > 0,
                masked_loss.sum() / mask_sum,
                jnp.array(0.0)  # Default value if mask_sum is 0
            )
            
            return loss
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            # Add labels back to batch
            batch["labels"] = labels
            if dropout_rng is not None:
                batch["dropout_rng"] = dropout_rng
            raise
    
    def _train_step(self, state, batch, rng=None, grad_clip_norm=1.0):
        """Single training step with gradient clipping and NaN detection"""
        try:
            # Handle PRNG keys for dropout
            if self.use_rng_keys_for_dropout and rng is None:
                # Create a new PRNG key if none is provided
                rng = jax.random.PRNGKey(int(time.time() * 1000) % (2**32))
            
            # Create a function that includes PRNG key handling
            def loss_fn_with_dropout(params, batch, dropout_rng):
                # If using RNG keys for dropout, add it to the batch
                if self.use_rng_keys_for_dropout:
                    # Add dropout_rng to the arguments
                    batch = dict(batch)  # Make a copy to avoid modifying the original
                    batch["dropout_rng"] = dropout_rng
                return self._loss_fn(params, batch)
            
            # Create value and grad function
            grad_fn = jax.value_and_grad(loss_fn_with_dropout)
            
            if self.use_rng_keys_for_dropout:
                # Split the RNG key for this step
                dropout_rng, new_rng = jax.random.split(rng)
                # Call with dropout RNG
                loss, grads = grad_fn(state.params, batch, dropout_rng)
            else:
                # Call without dropout RNG
                loss, grads = grad_fn(state.params, batch, None)
            
            # Check for NaNs in gradients
            if jnp.isnan(loss).any() or jnp.isinf(loss).any():
                logger.warning(f"NaN or Inf loss detected: {loss}")
                # Use a large but finite value instead of NaN
                loss = jnp.nan_to_num(loss, nan=1e3, posinf=1e3, neginf=1e3)
                return state, loss, new_rng if self.use_rng_keys_for_dropout else None  # Skip update
            
            # Check for NaNs in gradients - can be expensive, so we use a sample
            # Random sample some gradients to check for NaNs
            grad_flat, _ = jax.tree_util.tree_flatten(grads)
            for g in grad_flat[:5]:  # Check just a sample of gradients
                if jnp.isnan(g).any() or jnp.isinf(g).any():
                    logger.warning("NaN or Inf gradients detected, skipping update")
                    return state, loss, new_rng if self.use_rng_keys_for_dropout else None  # Skip update
            
            # Clip gradients to prevent explosions
            grads = jax.tree_util.tree_map(
                lambda g: jnp.clip(g, -1.0, 1.0),
                grads
            )
            
            # Apply gradients
            new_state = state.apply_gradients(grads=grads)
            return new_state, loss, new_rng if self.use_rng_keys_for_dropout else None
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            return state, jnp.array(float('nan')), rng
    
    def fine_tune(self, pruned_params, num_epochs=1, learning_rate=5e-5, evaluate_interval=5):
        """Fine-tune the pruned model"""
        logger.info(f"\nFine-tuning model with {self.dataset_name} dataset for {num_epochs} epochs...")
        
        # Reduce learning rate for larger models
        model_name = self.pruning_module.model_name.lower()
        if "large" in model_name or "1.3b" in model_name:
            learning_rate = learning_rate / 2
            logger.info(f"Reduced learning rate to {learning_rate} for large model")
        
        # Prepare dataset
        dataset = self._prepare_dataset()
        
        # Create training state
        self.train_state = self._create_train_state(pruned_params, learning_rate)
        self.metrics_history = []
        
        # Training loop
        total_steps = 0
        perplexity_history = []
        
        # Initialize PRNG key for dropout
        rng = jax.random.PRNGKey(int(time.time() * 1000) % (2**32)) if self.use_rng_keys_for_dropout else None
        
        for epoch in range(num_epochs):
            # Shuffled dataset for each epoch (if it's a list of batches)
            if isinstance(dataset, list):
                np.random.shuffle(dataset)
                epoch_dataset = dataset
            else:
                # If it's a datasets.Dataset, shuffle
                epoch_dataset = dataset.shuffle()
            
            # Create progress bar
            epoch_desc = f"Epoch {epoch+1}/{num_epochs}"
            batch_count = len(epoch_dataset) if hasattr(epoch_dataset, "__len__") else "?"
            progress_bar = tqdm(enumerate(epoch_dataset), desc=epoch_desc, 
                               total=batch_count if batch_count != "?" else None)
            
            epoch_losses = []
            nan_count = 0  # Track NaN losses
            
            for step, batch in progress_bar:
                # Train step
                try:
                    self.train_state, loss, new_rng = self._train_step(self.train_state, batch, rng)
                    # Update RNG for next step if using dropout
                    if self.use_rng_keys_for_dropout:
                        rng = new_rng
                        
                    total_steps += 1
                    
                    # Check for NaN loss
                    if jnp.isnan(loss).any() or jnp.isinf(loss).any():
                        nan_count += 1
                        logger.warning(f"NaN loss at step {total_steps}")
                        
                        # If we get too many NaNs, reduce batch size and learning rate
                        if nan_count > 5:
                            logger.warning("Too many NaN losses, reducing batch size and learning rate")
                            old_batch_size = self.batch_size
                            self.batch_size = max(1, self.batch_size // 2)
                            if old_batch_size == self.batch_size:  # Can't reduce further
                                logger.error("Cannot reduce batch size further, aborting fine-tuning")
                                break
                                
                            # Create new dataset with smaller batch size
                            dataset = self._prepare_dataset()
                            
                            # Create new train state with smaller learning rate
                            learning_rate = learning_rate / 2
                            logger.info(f"Reducing batch size to {self.batch_size} and learning rate to {learning_rate}")
                            self.train_state = self._create_train_state(self.train_state.params, learning_rate)
                            
                            # Reset epoch and start again
                            nan_count = 0
                            break
                        
                        continue  # Skip this step but keep going
                    
                    epoch_losses.append(loss.item())
                    
                    # Update progress bar
                    progress_bar.set_description(f"{epoch_desc} - Loss: {loss.item():.4f}")
                    
                    # Evaluate periodically
                    if total_steps % evaluate_interval == 0:
                        # Generate dummy text to check progress
                        prompt = "Artificial intelligence will transform"
                        try:
                            generated = self.pruning_module.generate_text(
                                self.train_state.params, prompt, max_length=30
                            )
                            perplexity = self.pruning_module.evaluate_perplexity(
                                self.train_state.params, prompt
                            )
                            perplexity_history.append((total_steps, perplexity))
                            logger.info(f"\nStep {total_steps} - Perplexity: {perplexity:.4f}")
                            logger.info(f"Generated: {generated}")
                        except Exception as e:
                            logger.error(f"Error evaluating model: {e}")
                except Exception as e:
                    logger.error(f"Error in training step: {e}")
                    # Continue to next batch
                    continue
                
                # Perform garbage collection every 10 steps to free memory
                if step % 10 == 0:
                    gc.collect()
            
            # End of epoch metrics
            epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('nan')
            logger.info(f"\nEpoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
            
            self.metrics_history.append({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "perplexity_history": perplexity_history,
                "nan_count": nan_count
            })
        
        logger.info("\nFine-tuning completed!")
        return self.train_state.params, self.metrics_history
    
    def plot_training_progress(self, figsize=(12, 10)):
        """Plot training progress"""
        if not self.metrics_history:
            logger.info("No training metrics available yet")
            return
        
        import matplotlib.pyplot as plt
        
        # Reset matplotlib parameters to ensure clean styling
        plt.rcParams.update(plt.rcParamsDefault)
        
        # Set better styling parameters
        plt.rcParams.update({
            'figure.figsize': figsize,
            'figure.titlesize': 14,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 9,
            'font.family': 'sans-serif'
        })
        
        # Extract epoch losses
        epochs = [m["epoch"] for m in self.metrics_history]
        losses = [m["loss"] for m in self.metrics_history]
        
        # Extract perplexity history
        steps = []
        perplexities = []
        for m in self.metrics_history:
            for step, perplexity in m.get("perplexity_history", []):
                steps.append(step)
                perplexities.append(perplexity)
        
        # Create figure with appropriate spacing
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Plot losses
        ax1 = axes[0]
        ax1.plot(epochs, losses, "o-", color="blue", linewidth=2, markersize=8)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, linestyle="--", alpha=0.7)
        
        # Make sure y-axis includes zero for better perspective
        ymin, ymax = ax1.get_ylim()
        if ymin > 0:
            ax1.set_ylim(bottom=0)
            
        # Add padding to x-axis for better display
        xmin, xmax = ax1.get_xlim()
        ax1.set_xlim(xmin - 0.2, xmax + 0.2)
        
        # Plot perplexities
        ax2 = axes[1]
        if steps and perplexities:
            ax2.plot(steps, perplexities, "o-", color="green", linewidth=2, markersize=8)
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Perplexity")
            ax2.set_title("Perplexity During Training")
            ax2.grid(True, linestyle="--", alpha=0.7)
            
            # Add padding to x-axis
            xmin, xmax = ax2.get_xlim()
            ax2.set_xlim(xmin - 0.5, xmax + 0.5)
        else:
            ax2.text(0.5, 0.5, "No perplexity data available",
                    ha="center", va="center", fontsize=12)
        
        # Plot NaN counts
        ax3 = axes[2]
        nan_counts = [m.get("nan_count", 0) for m in self.metrics_history]
        
        # Check if we have any NaN counts
        if any(nan_counts):
            ax3.bar(epochs, nan_counts, color="red", alpha=0.7)
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("NaN Count")
            ax3.set_title("NaN Losses per Epoch")
            ax3.grid(True, linestyle="--", alpha=0.7)
            
            # Add padding to x-axis
            xmin, xmax = ax3.get_xlim()
            ax3.set_xlim(xmin - 0.2, xmax + 0.2)
            
            # Ensure y-axis ticks are integers
            from matplotlib.ticker import MaxNLocator
            ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            # If no NaNs, show a message
            ax3.text(0.5, 0.5, "No NaN losses detected (good!)",
                    ha="center", va="center", fontsize=12, color="green")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("NaN Count")
            ax3.set_title("NaN Losses per Epoch")
        
        # Adjust layout with proper spacing
        fig.tight_layout(pad=2.0)
        plt.subplots_adjust(hspace=0.3)
        plt.show()
        
        return fig