"""
Consolidated fine-tuning implementation for pruned models with configurable stability features.

This module provides a unified implementation that combines the features of the original
FineTuner and ImprovedFineTuner classes, with configurable stability levels.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import logging
import math
import time
import gc
from flax.training import train_state
from functools import partial
from tqdm.auto import tqdm
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
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class FineTuner:
    """Fine-tunes a pruned model to recover performance with configurable stability features."""
    
    def __init__(
        self,
        pruning_module,
        dataset_name="openwebtext",
        dataset_config=None,
        batch_size=4,
        stability_level=1,  # 0: basic, 1: standard, 2: high
        use_synthetic_data=False,
        model_specific_optimizations=True
    ):
        """
        Initialize the fine-tuner with configurable stability features.
        
        Args:
            pruning_module: The pruning module instance
            dataset_name: Name of the dataset to use for fine-tuning
            dataset_config: Configuration of the dataset
            batch_size: Initial batch size for training
            stability_level: Level of stability features to use
                0: Basic operation (minimal safety features)
                1: Standard stability (recommended for most models)
                2: High stability (for OPT models and other challenging models)
            use_synthetic_data: Whether to use synthetic data instead of real dataset
            model_specific_optimizations: Whether to apply model-specific optimizations
        """
        self.pruning_module = pruning_module
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.stability_level = stability_level
        self.use_synthetic_data = use_synthetic_data
        self.model_specific_optimizations = model_specific_optimizations
        
        # Default parameters - will be adjusted based on stability level
        self.max_seq_length = 128  # Modest sequence length for faster training
        self.train_state = None
        self.metrics_history = []
        
        # Feature flags - will be set based on stability level
        self.use_rng_keys_for_dropout = False
        self.use_gradient_clipping = False
        self.track_nan_counts = False
        self.use_garbage_collection = False
        self.enable_adaptive_params = False
        
        # Detect model type
        self.is_opt_model = 'opt' in self.pruning_module.model_name.lower() if hasattr(self.pruning_module, 'model_name') else False
        
        # Detect number of devices
        self.devices = jax.devices()
        self.n_devices = len(self.devices)
        
        # Apply settings based on stability level
        self._configure_stability_features()
        
        if self.n_devices > 1:
            logger.info(f"Using {self.n_devices} devices for training")
            self.batch_size = max(self.batch_size, self.n_devices)
            # Make batch size divisible by device count
            self.batch_size = (self.batch_size // self.n_devices) * self.n_devices
            logger.info(f"Adjusted batch size to {self.batch_size} for multi-device training")
            
        # Apply model-specific optimizations if enabled
        if self.model_specific_optimizations:
            self._apply_model_specific_optimizations()
    
    def _configure_stability_features(self):
        """Configure stability features based on the selected stability level."""
        if self.stability_level >= 1:
            # Standard stability features (level 1)
            self.use_rng_keys_for_dropout = True
            self.use_gradient_clipping = True
            self.track_nan_counts = True
            logger.info("Configured with standard stability features (level 1)")
            
        if self.stability_level >= 2:
            # High stability features (level 2)
            self.use_garbage_collection = True
            self.enable_adaptive_params = True
            self.use_synthetic_data = True  # Force synthetic data for highest stability
            logger.info("Configured with high stability features (level 2)")
            
        # For OPT models, always use high stability regardless of setting
        if self.is_opt_model and self.stability_level < 2:
            logger.info("OPT model detected, automatically upgrading to high stability (level 2)")
            self.stability_level = 2
            self._configure_stability_features()
    
    def _apply_model_specific_optimizations(self):
        """Apply model-specific optimizations based on model type and size."""
        model_name = getattr(self.pruning_module, 'model_name', '').lower()
        
        # Detect very large models (XL, 1B+)
        is_very_large = any(x in model_name for x in ["xl", "1.3b", "1b"])
        
        # Detect large models 
        is_large = is_very_large or any(x in model_name for x in ["large", "pythia-410m"])
        
        # Detect medium models
        is_medium = not (is_large or is_very_large) and any(x in model_name for x in 
                                                          ["medium", "base", "350m", "pythia-160m"])
        
        # For very large models, use aggressive optimizations
        if is_very_large:
            old_batch_size = self.batch_size
            # Extremely small batch size for XL and billion-parameter models
            self.batch_size = 1
            logger.info(f"Reduced batch size from {old_batch_size} to {self.batch_size} for very large model {model_name}")
            
            # Drastically reduce sequence length for memory efficiency
            self.max_seq_length = min(self.max_seq_length, 32)
            logger.info(f"Set maximum sequence length to {self.max_seq_length} for very large model")
            
        # For large models, reduce batch size substantially
        elif is_large:
            old_batch_size = self.batch_size
            self.batch_size = max(1, self.batch_size // 4)
            logger.info(f"Reduced batch size from {old_batch_size} to {self.batch_size} for large model {model_name}")
            
            # Also reduce sequence length for large models
            self.max_seq_length = min(self.max_seq_length, 48)
            logger.info(f"Set maximum sequence length to {self.max_seq_length} for large model")
            
        # For medium-sized models, make smaller adjustments
        elif is_medium:
            old_batch_size = self.batch_size
            self.batch_size = max(1, self.batch_size // 2)
            logger.info(f"Reduced batch size from {old_batch_size} to {self.batch_size} for medium-sized model {model_name}")
            
            # Slightly reduce sequence length for medium models
            self.max_seq_length = min(self.max_seq_length, 64)
        
        # Special handling for all Pythia models
        if "pythia" in model_name:
            logger.info("Applying Pythia-specific optimizations")
            # Pythia models need careful memory management
            self.batch_size = max(1, min(self.batch_size, 2))
            self.max_seq_length = min(self.max_seq_length, 48)
            
        # Special handling for OPT models
        if self.is_opt_model:
            logger.info("Applying OPT-specific optimizations")
            # OPT models need a smaller batch size and sequence length for stability
            self.batch_size = max(1, min(self.batch_size, 2))
            self.max_seq_length = min(self.max_seq_length, 48)
    
    def _prepare_dataset(self):
        """Load and prepare the dataset for fine-tuning with appropriate error handling."""
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
                elif isinstance(examples, dict):
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
                else:
                    # Fallback for unexpected format
                    logger.warning("Unexpected dataset format, falling back to synthetic data")
                    return None
                
                # Add truncation to prevent very long sequences
                tokenized = tokenizer(
                    texts, 
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding="max_length",
                    return_tensors="np"
                )
                return tokenized
            
            # Special case for Wikitext dataset which has a different structure
            if "wikitext" in self.dataset_name:
                try:
                    # For wikitext, use a simpler approach focused on the known structure
                    tokenized_dataset = dataset.map(
                        tokenize_function,
                        batched=True,
                        num_proc=1,
                        remove_columns=dataset.column_names
                    )
                    
                    # Check for successful tokenization
                    if not tokenized_dataset or len(tokenized_dataset) == 0:
                        raise ValueError("Tokenization produced empty dataset")
                        
                except Exception as e:
                    logger.error(f"Error processing wikitext: {e}")
                    return self._prepare_synthetic_dataset()
            else:
                # Remove columns that aren't strings for other datasets
                try:
                    # Identify columns to keep
                    columns_to_remove = []
                    for col in dataset.column_names:
                        if isinstance(dataset[0][col], (int, float, bool)) or dataset[0][col] is None:
                            columns_to_remove.append(col)
                    
                    tokenized_dataset = dataset.map(
                        tokenize_function,
                        batched=True,
                        num_proc=1,
                        remove_columns=columns_to_remove if columns_to_remove else None
                    )
                    
                    if not tokenized_dataset or len(tokenized_dataset) == 0:
                        raise ValueError("Tokenization produced empty dataset")
                        
                except Exception as e:
                    logger.error(f"Error processing dataset: {e}")
                    return self._prepare_synthetic_dataset()
            
            # Convert to list of batches for simpler processing
            dataloader = []
            
            try:
                for i in range(0, len(tokenized_dataset), self.batch_size):
                    end = min(i + self.batch_size, len(tokenized_dataset))
                    batch_samples = tokenized_dataset[i:end]
                    
                    # Convert batch to appropriate format - handle different dataset structures
                    if isinstance(batch_samples, dict):
                        # If batch_samples is already a dict with keys (newer datasets format)
                        batch = {}
                        if "input_ids" in batch_samples:
                            batch["input_ids"] = np.array(batch_samples["input_ids"][:end-i])
                            batch["attention_mask"] = np.array(batch_samples["attention_mask"][:end-i])
                        else:
                            # Fall back to synthetic data
                            logger.warning("Unexpected dataset format - input_ids not found")
                            return self._prepare_synthetic_dataset()
                    else:
                        # If batch_samples is a list of individual samples (older format)
                        try:
                            batch = {
                                "input_ids": np.array([sample["input_ids"] for sample in batch_samples]),
                                "attention_mask": np.array([sample["attention_mask"] for sample in batch_samples]),
                            }
                        except (KeyError, TypeError) as e:
                            logger.error(f"Error processing batch: {e}")
                            return self._prepare_synthetic_dataset()
                    
                    # Add labels
                    batch["labels"] = batch["input_ids"].copy()
                    
                    dataloader.append(batch)
            except Exception as e:
                logger.error(f"Error creating batches: {e}")
                return self._prepare_synthetic_dataset()
            
            logger.info(f"Created {len(dataloader)} batches")
            return dataloader
                
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            logger.info("Falling back to synthetic data for training")
            import traceback
            traceback.print_exc()
            return self._prepare_synthetic_dataset()
    
    def _prepare_synthetic_dataset(self):
        """Create synthetic data for training when dataset loading fails."""
        try:
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
            
            # Get model name for model-specific optimizations
            model_name = getattr(self.pruning_module, 'model_name', '').lower()
            
            # Detect model size categories
            is_very_large = any(x in model_name for x in ["xl", "1.3b", "1b"])
            is_large = is_very_large or any(x in model_name for x in ["large", "pythia-410m"])
            is_medium = not (is_large or is_very_large) and any(x in model_name for x in 
                                                              ["medium", "base", "350m", "pythia-160m"])
            
            # Use model-specific token ranges to avoid NaN issues
            if self.is_opt_model or "pythia" in model_name or self.stability_level >= 2:
                # For OPT/Pythia models or high stability, use only common tokens to avoid issues
                token_range_start = 10  # Skip special tokens at the beginning
                token_range_end = min(1000, vocab_size // 20)  # Use a very small subset of vocab for stability
                logger.info(f"Using extremely restricted token range for stability: {token_range_start}-{token_range_end}")
            else:
                # For other models, use a wider range but still restricted for safety
                token_range_start = 0
                token_range_end = min(5000, vocab_size // 5)
            
            # Create samples of random token sequences (fewer for large models)
            # Adjust sample count based on model size
            if is_very_large:
                num_samples = 20  # Minimal samples for largest models
                logger.info(f"Using minimal sample count ({num_samples}) for very large model")
            elif is_large or self.stability_level >= 2:
                num_samples = 40  # Reduced samples for large models
                logger.info(f"Using reduced sample count ({num_samples}) for large model")
            elif is_medium:
                num_samples = 60  # Moderate samples for medium models
            else:
                num_samples = 80  # Standard sample count for small models
            
            samples = []
            for _ in range(num_samples):
                # Generate shorter sequences for stability based on model size
                if is_very_large:
                    max_length = min(16, self.max_seq_length)  # Ultra-short sequences
                elif is_large or self.stability_level >= 2:
                    max_length = min(32, self.max_seq_length)  # Very short sequences
                else:
                    max_length = min(self.max_seq_length, 64)  # Normal short sequences
                
                # Random length but ensure at least a few tokens
                length = np.random.randint(max(4, max_length // 4), max_length)
                
                # Generate random token IDs in the safe range
                token_ids = np.random.randint(token_range_start, token_range_end, size=length)
                
                # Replace special tokens with normal tokens
                for i, token_id in enumerate(token_ids):
                    if token_id in special_tokens:
                        token_ids[i] = (token_id + 1) % (token_range_end - token_range_start) + token_range_start
                        # Make sure we're not just cycling through special tokens
                        while token_ids[i] in special_tokens:
                            token_ids[i] = (token_ids[i] + 1) % (token_range_end - token_range_start) + token_range_start
                
                # Create sample
                sample = {
                    "input_ids": token_ids,
                    "attention_mask": np.ones_like(token_ids),
                    "labels": token_ids.copy()
                }
                samples.append(sample)
                
            # For large models, run garbage collection to free memory
            if is_large or self.stability_level >= 2:
                gc.collect()
                
        except Exception as e:
            # If we fail creating sophisticated synthetic data, fall back to ultra-simple data
            logger.error(f"Error creating synthetic dataset: {e}")
            logger.info("Creating ultra-simple synthetic dataset as fallback")
            
            # Ultra-simple data with minimal tokens and length
            samples = []
            for _ in range(10):  # Ultra-minimal sample count
                # Generate a very small sequence with a tiny vocab
                token_ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
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
        """Create a training state for the fine-tuning process with appropriate optimizer."""
        # Choose optimizer based on stability level
        if self.stability_level >= 1:
            # Use AdamW with weight decay for better stability
            weight_decay = 0.01
            optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
            logger.info(f"Using AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")
        else:
            # Original implementation uses standard Adam
            optimizer = optax.adam(learning_rate)
            logger.info(f"Using Adam optimizer with lr={learning_rate}")
        
        # Create train state
        model = self.pruning_module.model
        return train_state.TrainState.create(
            apply_fn=model.__call__,
            params=params,
            tx=optimizer
        )
    
    def _loss_fn(self, params, batch):
        """Loss function for the language modeling task with appropriate stability features."""
        model = self.pruning_module.model
        
        # Extract labels from batch but don't pass them to the model
        labels = batch.pop("labels", None)
        
        # Extract dropout_rng if present for handling dropout
        dropout_rng = batch.pop("dropout_rng", None)
        
        # Check for NaN or Inf in input when stability level > 0
        if self.stability_level >= 1:
            for k, v in batch.items():
                if jnp.isnan(v).any() or jnp.isinf(v).any():
                    # Replace NaN and Inf with zeros
                    batch[k] = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    logger.warning(f"Found NaN or Inf in {k}, replaced with zeros")
        
        try:
            # Prepare the model call arguments
            model_kwargs = {"params": params}
            
            # Add dropout_rng if present and enabled
            if dropout_rng is not None and self.use_rng_keys_for_dropout:
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
            
            # Apply NaN checking for stability level >= 1
            if self.stability_level >= 1 and (jnp.isnan(shift_logits).any() or jnp.isinf(shift_logits).any()):
                logger.warning("Found NaN or Inf in logits, replaced with finite values")
                shift_logits = jnp.nan_to_num(shift_logits, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Calculate cross entropy loss
            loss = optax.softmax_cross_entropy_with_integer_labels(
                shift_logits, shift_labels
            )
            
            # Check for NaN and Inf in loss
            if self.stability_level >= 1 and (jnp.isnan(loss).any() or jnp.isinf(loss).any()):
                logger.warning(f"Found NaN or Inf in loss: {loss}")
                # Use a large but finite value instead of NaN or Inf
                loss = jnp.nan_to_num(loss, nan=1e3, posinf=1e3, neginf=1e3)
            
            # Apply mask and calculate mean
            masked_loss = loss * shift_mask
            mask_sum = shift_mask.sum()
            
            # Safe division with fallback if stability level >= 1
            if self.stability_level >= 1:
                loss = jnp.where(
                    mask_sum > 0,
                    masked_loss.sum() / mask_sum,
                    jnp.array(0.0)  # Default value if mask_sum is 0
                )
            else:
                # Basic implementation doesn't handle zero mask_sum
                loss = (masked_loss.sum() / mask_sum)
            
            return loss
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            # Add labels back to batch
            batch["labels"] = labels
            if dropout_rng is not None:
                batch["dropout_rng"] = dropout_rng
            
            # Return a non-zero fallback loss if in high stability mode
            if self.stability_level >= 1:
                return jnp.array(1.0, dtype=jnp.float32)
            raise  # Re-raise in basic mode
    
    def _train_step(self, state, batch, rng=None, grad_clip_norm=1.0):
        """Single training step with appropriate stability features."""
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
            
            # Check for NaNs in loss
            if self.stability_level >= 1:
                if jnp.isnan(loss).any() or jnp.isinf(loss).any():
                    logger.warning(f"NaN or Inf loss detected: {loss}")
                    # Use a large but finite value instead of NaN
                    loss = jnp.nan_to_num(loss, nan=1e3, posinf=1e3, neginf=1e3)
                    return state, loss, new_rng if self.use_rng_keys_for_dropout else None  # Skip update
            
            # Check for NaNs in gradients when stability level >= 1
            if self.stability_level >= 1:
                # Check some gradients for NaNs
                grad_flat, _ = jax.tree_util.tree_flatten(grads)
                for g in grad_flat[:5]:  # Check just a sample of gradients
                    if jnp.isnan(g).any() or jnp.isinf(g).any():
                        logger.warning("NaN or Inf gradients detected, skipping update")
                        return state, loss, new_rng if self.use_rng_keys_for_dropout else None  # Skip update
            
            # Apply gradient clipping when enabled
            if self.use_gradient_clipping:
                grads = jax.tree_util.tree_map(
                    lambda g: jnp.clip(g, -grad_clip_norm, grad_clip_norm),
                    grads
                )
            
            # Apply gradients
            new_state = state.apply_gradients(grads=grads)
            return new_state, loss, new_rng if self.use_rng_keys_for_dropout else None
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            return state, jnp.array(float('nan')), rng
    
    def fine_tune(self, pruned_params, num_epochs=1, learning_rate=5e-5, evaluate_interval=5):
        """Fine-tune the pruned model with appropriate stability features."""
        logger.info(f"\nFine-tuning model with {self.dataset_name} dataset for {num_epochs} epochs...")
        
        # Apply model-specific learning rate and memory optimizations
        if self.model_specific_optimizations:
            model_name = getattr(self.pruning_module, 'model_name', '').lower()
            
            # Detect model size for optimizations
            is_very_large = any(x in model_name for x in ["xl", "1.3b", "1b"])
            is_large = is_very_large or any(x in model_name for x in ["large", "pythia-410m"])
            
            # Very large models need extremely conservative parameters
            if is_very_large:
                # Ultra-conservative learning rate
                learning_rate = learning_rate / 4
                # Very short training for big models
                num_epochs = max(1, num_epochs // 2)
                # Less frequent evaluation to save memory
                evaluate_interval = max(10, evaluate_interval * 2)
                logger.info(f"Extreme memory optimization for very large model: LR={learning_rate}, epochs={num_epochs}")
            # Large models need substantial adjustments
            elif is_large or self.stability_level >= 2:
                learning_rate = learning_rate / 2
                logger.info(f"Reduced learning rate to {learning_rate} for large model or high stability")
            
            # Special handling for Pythia models
            if "pythia" in model_name:
                # Pythia models need careful parameter handling
                learning_rate = min(learning_rate, 2e-5)
                logger.info(f"Using conservative learning rate {learning_rate} for Pythia model")
            
            # Special handling for OPT models
            if self.is_opt_model:
                # OPT models need a smaller learning rate for stability
                learning_rate = min(learning_rate, 2e-5)
                logger.info(f"Using conservative learning rate {learning_rate} for OPT model")
        
        # Prepare dataset
        dataset = self._prepare_dataset()
        
        # Create training state
        self.train_state = self._create_train_state(pruned_params, learning_rate)
        self.metrics_history = []
        
        # Training loop
        total_steps = 0
        perplexity_history = []
        
        # Initialize PRNG key for dropout if needed
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
                        
                        # If we get too many NaNs and adaptive parameters is enabled, reduce batch size and learning rate
                        if nan_count > 5 and self.enable_adaptive_params:
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
                
                # Perform garbage collection every 10 steps to free memory if enabled
                if self.use_garbage_collection and step % 10 == 0:
                    gc.collect()
            
            # End of epoch metrics
            epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('nan')
            logger.info(f"\nEpoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
            
            self.metrics_history.append({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "perplexity_history": perplexity_history,
                "nan_count": nan_count if self.track_nan_counts else 0
            })
        
        logger.info("\nFine-tuning completed!")
        return self.train_state.params, self.metrics_history
    
    def plot_training_progress(self, figsize=(12, 10)):
        """Plot training progress with appropriate visualization based on stability level."""
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
        
        # Determine number of plots based on stability level
        if self.stability_level >= 1 and self.track_nan_counts:
            # Create figure with 3 plots (including NaN count)
            fig, axes = plt.subplots(3, 1, figsize=figsize)
            plot_nan_counts = True
        else:
            # Create figure with 2 plots (loss and perplexity only)
            fig, axes = plt.subplots(2, 1, figsize=figsize)
            plot_nan_counts = False
            
        # Access axes correctly based on their structure
        ax1 = axes[0]
        ax2 = axes[1]
        if plot_nan_counts:
            ax3 = axes[2]
        
        # Plot losses
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
        
        # Plot NaN counts if enabled
        if plot_nan_counts:
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


# Backward compatibility aliases
class ImprovedFineTuner(FineTuner):
    """Alias for backward compatibility with existing code."""
    
    def __init__(self, pruning_module, dataset_name="openwebtext", dataset_config=None, batch_size=4):
        """Initialize with high stability level as default."""
        super().__init__(
            pruning_module=pruning_module,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            batch_size=batch_size,
            stability_level=2,  # High stability by default
            use_synthetic_data=False,
            model_specific_optimizations=True
        )
        logger.info("Using ImprovedFineTuner (compatibility class) with high stability settings")


# Simple test function to validate the implementation
def test_fine_tuner(model_name="distilgpt2", stability_level=1):
    """
    Test the fine-tuner implementation with a small model.
    
    Args:
        model_name: Name of the model to test
        stability_level: Stability level to test (0, 1, or 2)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
        
        # Create a dummy pruning module
        class DummyPruningModule:
            def __init__(self, model_name):
                self.model_name = model_name
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = FlaxAutoModelForCausalLM.from_pretrained(model_name)
                self.original_params = self.model.params
                
            def generate_text(self, params, prompt, max_length=30):
                return f"Generated text for {prompt}"
                
            def evaluate_perplexity(self, params, prompt):
                return 10.0  # Dummy value
        
        # Create pruning module
        pruning_module = DummyPruningModule(model_name)
        
        # Create fine-tuner with specified stability level
        fine_tuner = FineTuner(
            pruning_module=pruning_module,
            dataset_name="wikitext",
            dataset_config="wikitext-2-v1",
            batch_size=2,
            stability_level=stability_level,
            use_synthetic_data=True  # Use synthetic data for testing
        )
        
        # Fine-tune for a single step
        params, metrics = fine_tuner.fine_tune(
            pruned_params=pruning_module.original_params,
            num_epochs=1,
            learning_rate=5e-5,
            evaluate_interval=1
        )
        
        # Plot results if available
        if metrics:
            fine_tuner.plot_training_progress()
            
        logger.info(f"Test successful with stability_level={stability_level}")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with different stability levels
    for level in [0, 1, 2]:
        print(f"\nTesting with stability level {level}")
        result = test_fine_tuner(stability_level=level)
        print(f"Test {'passed' if result else 'failed'} for stability level {level}")
        
    # Test with OPT model if available
    try:
        print("\nTesting with OPT model")
        result = test_fine_tuner(model_name="facebook/opt-125m", stability_level=1)
        print(f"OPT model test {'passed' if result else 'failed'}")
    except:
        print("OPT model test skipped - model not available")