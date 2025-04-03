"""
Fine-tuning implementation for pruned models using JAX/Flax
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import logging
from flax.training import train_state
from functools import partial
from tqdm.auto import tqdm

# Set up logging
logger = logging.getLogger(__name__)

# Import datasets correctly to avoid conflict with local module
try:
    from datasets import load_dataset
except ImportError:
    # Handle the case where huggingface datasets is not available
    # or there's a naming conflict with a local module
    import importlib.util
    if importlib.util.find_spec("huggingface_hub"):
        from huggingface_hub import hf_hub_download
        def load_dataset(*args, **kwargs):
            logger.warning("Using placeholder load_dataset function")
            return None
    else:
        logger.warning("datasets module not found, synthetic data will be used")
        def load_dataset(*args, **kwargs):
            return None

from transformers import FlaxAutoModelForCausalLM


class FineTuner:
    """Fine-tunes a pruned model to recover performance"""
    
    def __init__(self, pruning_module, dataset_name="openwebtext", batch_size=4):
        self.pruning_module = pruning_module
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_seq_length = 128  # Modest sequence length for faster training
        self.train_state = None
        self.metrics_history = []
        
        # Detect number of devices
        self.devices = jax.devices()
        self.n_devices = len(self.devices)
        if self.n_devices > 1:
            print(f"Using {self.n_devices} devices for training")
            self.batch_size = max(self.batch_size, self.n_devices)
            # Make batch size divisible by device count
            self.batch_size = (self.batch_size // self.n_devices) * self.n_devices
            print(f"Adjusted batch size to {self.batch_size} for multi-device training")
    
    def _prepare_dataset(self):
        """Load and prepare the dataset for fine-tuning"""
        try:
            # Try to load a small portion of the dataset for faster loading
            dataset = load_dataset(self.dataset_name, split="train[:5000]")
            
            # Process dataset
            tokenizer = self.pruning_module.tokenizer
            
            def tokenize_function(examples):
                # Tokenize the texts
                tokenized = tokenizer(examples["text"])
                return tokenized
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                num_proc=1,
                remove_columns=["text"]
            )
            
            # Create data loader
            def create_batch(samples):
                # Prepare batch of appropriate shape
                batch = {k: np.array(v) for k, v in samples.items()}
                
                # Create 'labels' for the causal language modeling task
                batch["labels"] = batch["input_ids"].copy()
                
                # Get sequence lengths
                seq_lengths = (batch["input_ids"] != tokenizer.pad_token_id).sum(axis=1)
                
                # Loop through samples and pad/truncate as needed
                for i, length in enumerate(seq_lengths):
                    # Ensure we have at least 2 tokens (can't shift with just 1)
                    if length < 2:
                        # Add padding to have at least 2 tokens
                        padding = np.array([tokenizer.pad_token_id] * (2 - length))
                        batch["input_ids"][i] = np.concatenate([batch["input_ids"][i][:length], padding])
                        batch["attention_mask"][i] = np.concatenate([batch["attention_mask"][i][:length], 
                                                                    np.ones_like(padding)])
                        batch["labels"][i] = np.concatenate([batch["labels"][i][:length], padding])
                        seq_lengths[i] = 2
                    
                    # Truncate to max sequence length if needed
                    if length > self.max_seq_length:
                        batch["input_ids"][i] = batch["input_ids"][i][:self.max_seq_length]
                        batch["attention_mask"][i] = batch["attention_mask"][i][:self.max_seq_length]
                        batch["labels"][i] = batch["labels"][i][:self.max_seq_length]
                        seq_lengths[i] = self.max_seq_length
                
                return batch
            
            # Create data loader
            dataloader = tokenized_dataset.batch(self.batch_size)
            dataloader = dataloader.map(create_batch, batched=True)
            
            return dataloader
        
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            print("Falling back to synthetic data for training")
            return self._prepare_synthetic_dataset()
    
    def _prepare_synthetic_dataset(self):
        """Create synthetic data for training when dataset loading fails"""
        tokenizer = self.pruning_module.tokenizer
        
        # Generate random token IDs (avoid special tokens)
        vocab_size = tokenizer.vocab_size
        special_tokens = set([tokenizer.pad_token_id, tokenizer.eos_token_id, 
                             tokenizer.bos_token_id, tokenizer.unk_token_id])
        
        # Create 100 samples of random token sequences
        samples = []
        for _ in range(100):
            # Generate random length between 10 and max_seq_length
            length = np.random.randint(10, self.max_seq_length)
            
            # Generate random token IDs
            token_ids = np.random.randint(0, vocab_size, size=length)
            
            # Replace special tokens with normal tokens
            for i, token_id in enumerate(token_ids):
                if token_id in special_tokens:
                    token_ids[i] = (token_id + 1) % vocab_size
            
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
        
        return batches
    
    def _create_train_state(self, params, learning_rate=5e-5):
        """Create a training state for the fine-tuning process"""
        # Create optimizer
        optimizer = optax.adam(learning_rate)
        
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
        
        # Get logits from model
        outputs = model(**batch, params=params, train=True)
        logits = outputs.logits
        
        # Get labels and create masks
        labels = batch["labels"]
        
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
        
        # Apply mask and calculate mean
        loss = (loss * shift_mask).sum() / shift_mask.sum()
        
        return loss
    
    @partial(jax.jit, static_argnums=(0,))
    def _train_step(self, state, batch):
        """Single training step"""
        grad_fn = jax.value_and_grad(self._loss_fn)
        loss, grads = grad_fn(state.params, batch)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss
    
    def fine_tune(self, pruned_params, num_epochs=1, learning_rate=5e-5, evaluate_interval=5):
        """Fine-tune the pruned model"""
        print(f"\nFine-tuning model with {self.dataset_name} dataset for {num_epochs} epochs...")
        
        # Prepare dataset
        dataset = self._prepare_dataset()
        
        # Create training state
        self.train_state = self._create_train_state(pruned_params, learning_rate)
        self.metrics_history = []
        
        # Training loop
        total_steps = 0
        perplexity_history = []
        
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
            
            for step, batch in progress_bar:
                # Train step
                self.train_state, loss = self._train_step(self.train_state, batch)
                total_steps += 1
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
                        print(f"\nStep {total_steps} - Perplexity: {perplexity:.4f}")
                        print(f"Generated: {generated}")
                    except Exception as e:
                        print(f"Error evaluating model: {e}")
            
            # End of epoch metrics
            epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            print(f"\nEpoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
            
            self.metrics_history.append({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "perplexity_history": perplexity_history
            })
        
        print("\nFine-tuning completed!")
        return self.train_state.params, self.metrics_history
    
    def plot_training_progress(self, figsize=(12, 6)):
        """Plot training progress"""
        if not self.metrics_history:
            print("No training metrics available yet")
            return
        
        import matplotlib.pyplot as plt
        
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
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot losses
        ax1.plot(epochs, losses, "o-", color="blue")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, linestyle="--", alpha=0.7)
        
        # Plot perplexities
        if steps and perplexities:
            ax2.plot(steps, perplexities, "o-", color="green")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Perplexity")
            ax2.set_title("Perplexity During Training")
            ax2.grid(True, linestyle="--", alpha=0.7)
        else:
            ax2.text(0.5, 0.5, "No perplexity data available",
                    ha="center", va="center", fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return fig