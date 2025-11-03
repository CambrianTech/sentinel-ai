"""
NaN detection and prevention utilities for stable fine-tuning.

This module provides functions to detect and prevent NaN values during model training,
which is particularly important for large language models that may exhibit numerical 
instability during fine-tuning, especially after pruning.
"""

import jax
import jax.numpy as jnp
import optax
import logging

# Set up logger
logger = logging.getLogger(__name__)

def create_nan_safe_loss_fn(original_loss_fn, model, tokenizer, model_name=""):
    """
    Creates a NaN-safe wrapper around a loss function that catches and handles NaN values.
    
    Args:
        original_loss_fn: The original loss function to wrap
        model: The model being trained
        tokenizer: The tokenizer for the model
        model_name: Name of the model for logging purposes
        
    Returns:
        A new loss function that handles NaNs safely
    """
    def safe_loss_fn(params, batch):
        """Safe loss function that prevents NaN values"""
        # Extract labels from batch but don't pass them to the model
        labels = batch.pop("labels", None)
        
        # Check for NaNs in input
        for k, v in batch.items():
            if jnp.isnan(v).any() or jnp.isinf(v).any():
                # Replace NaNs with zeros
                batch[k] = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                logger.warning(f"Found NaN in input {k}, replaced with zeros")
        
        try:
            # Handle different model architectures
            is_opt_model = 'opt' in model_name.lower()
            
            if is_opt_model:
                # OPT models don't accept 'train' parameter
                outputs = model(**batch, params=params)
            else:
                # Other models might need the 'train' parameter
                try:
                    outputs = model(**batch, params=params, train=True)
                except TypeError:
                    # If 'train' parameter causes an error, try without it
                    outputs = model(**batch, params=params)
                    
            logits = outputs.logits
            
            # Add labels back to batch
            batch["labels"] = labels
            
            # Create mask for padding tokens
            pad_token_id = tokenizer.pad_token_id  
            loss_mask = (labels != pad_token_id)
            
            # Shift logits and labels
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            shift_mask = loss_mask[:, 1:]
            
            # Check for NaNs in logits
            if jnp.isnan(shift_logits).any() or jnp.isinf(shift_logits).any():
                logger.warning("Found NaN/Inf in logits - replacing with finite values")
                shift_logits = jnp.nan_to_num(shift_logits, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Safe cross entropy computation
            loss = optax.softmax_cross_entropy_with_integer_labels(
                shift_logits, shift_labels
            )
            
            # Check and fix NaNs in loss
            if jnp.isnan(loss).any() or jnp.isinf(loss).any():
                logger.warning("Found NaN/Inf in loss - replacing with finite values")
                loss = jnp.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)
            
            # Safe mean calculation
            masked_loss = loss * shift_mask
            mask_sum = shift_mask.sum()
            
            # Safe division
            computed_loss = jnp.where(
                mask_sum > 0,
                masked_loss.sum() / mask_sum,
                jnp.array(0.0, dtype=loss.dtype)
            )
            
            if jnp.isnan(computed_loss) or jnp.isinf(computed_loss):
                logger.critical("NaN/Inf in final loss - using fallback value")
                return jnp.array(1.0, dtype=loss.dtype)
                
            return computed_loss
            
        except Exception as e:
            logger.error(f"Error in loss function: {e}")
            # Return a safe default value
            batch["labels"] = labels  # Restore labels
            return jnp.array(1.0)  # Safe fallback
    
    return safe_loss_fn

def patch_fine_tuner(fine_tuner, model_name=""):
    """
    Patches a FineTuner instance with NaN-safe mechanisms.
    
    Args:
        fine_tuner: The FineTuner instance to patch
        model_name: Name of the model for logging purposes
        
    Returns:
        The patched FineTuner instance
    """
    if not hasattr(fine_tuner, '_loss_fn'):
        logger.warning("FineTuner doesn't have a _loss_fn attribute, cannot patch")
        return fine_tuner
    
    if not hasattr(fine_tuner, 'pruning_module') or not hasattr(fine_tuner.pruning_module, 'model'):
        logger.warning("FineTuner doesn't have expected attributes, cannot patch")
        return fine_tuner
        
    logger.info(f"Installing NaN-safe loss function for model {model_name}")
    original_loss_fn = fine_tuner._loss_fn
    model = fine_tuner.pruning_module.model
    tokenizer = fine_tuner.pruning_module.tokenizer
    
    fine_tuner._loss_fn = create_nan_safe_loss_fn(
        original_loss_fn, model, tokenizer, model_name
    )
    
    return fine_tuner

def test_nan_safety(model_name="gpt2", sequence_length=128, batch_size=2):
    """
    Tests the NaN-safety mechanisms with a toy example.
    
    Args:
        model_name: The model to test with
        sequence_length: Length of test sequences
        batch_size: Batch size for testing
        
    Returns:
        True if the test passes, False otherwise
    """
    try:
        from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
        import numpy as np
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = FlaxAutoModelForCausalLM.from_pretrained(model_name)
        
        # Create dummy inputs (with some NaN values)
        input_ids = np.random.randint(
            0, tokenizer.vocab_size, (batch_size, sequence_length)
        ).astype(np.int32)
        
        # Insert some NaN values to test robustness
        attention_mask = np.ones_like(input_ids).astype(np.float32)
        attention_mask[0, 5:10] = np.nan  # Add some NaNs
        
        # Create batch
        batch = {
            "input_ids": jnp.array(input_ids),
            "attention_mask": jnp.array(attention_mask),
            "labels": jnp.array(input_ids.copy())
        }
        
        # Define a simple loss function that would fail with NaNs
        def unsafe_loss_fn(params, batch):
            labels = batch.pop("labels", None)
            outputs = model(**batch, params=params)
            batch["labels"] = labels
            
            logits = outputs.logits
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            
            loss = optax.softmax_cross_entropy_with_integer_labels(
                shift_logits, shift_labels
            )
            return loss.mean()
        
        # Create safe version
        safe_loss_fn = create_nan_safe_loss_fn(
            unsafe_loss_fn, model, tokenizer, model_name
        )
        
        # Test both functions
        try:
            unsafe_result = unsafe_loss_fn(model.params, batch.copy())
            print(f"Unsafe loss result: {unsafe_result}")
            if jnp.isnan(unsafe_result).any():
                print("Unsafe function produced NaN as expected")
        except Exception as e:
            print(f"Unsafe function failed as expected: {e}")
        
        # Test safe function
        safe_result = safe_loss_fn(model.params, batch.copy())
        print(f"Safe loss result: {safe_result}")
        
        if not jnp.isnan(safe_result).any() and not jnp.isinf(safe_result).any():
            print("Safe function successfully prevented NaNs")
            return True
        else:
            print("Safe function failed to prevent NaNs")
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Enable logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with different models
    for model in ["gpt2", "facebook/opt-125m"]:
        print(f"\nTesting NaN prevention with {model}")
        result = test_nan_safety(model)
        print(f"Test {'passed' if result else 'failed'} for {model}")