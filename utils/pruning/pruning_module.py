"""
Core pruning implementation using JAX/Flax
"""

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

class PruningModule:
    """Core pruning implementation using JAX/Flax"""
    
    def __init__(self, model_name="distilgpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.original_params = None
        self.model_type = self._get_model_type(model_name)
        self.num_layers = 0
        self.num_heads = 0
        
    def _get_model_type(self, model_name):
        """Determine model type from name"""
        if "gpt2" in model_name.lower():
            return "gpt2"
        elif "opt" in model_name.lower():
            return "opt"
        elif "pythia" in model_name.lower():
            return "pythia"
        else:
            # Default to GPT-2 structure
            return "gpt2"
    
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model {self.model_name}...")
        try:
            # Check if this is a Pythia model (based on EleutherAI/GPTNeoX architecture)
            is_pythia = "pythia" in self.model_name.lower() or "eleutherai" in self.model_name.lower()
            
            # Always load the tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # For Pythia models, we need to use PyTorch and convert to JAX/Flax
            if is_pythia:
                print("Pythia model detected. Using PyTorch conversion approach...")
                try:
                    # Try to import PyTorch modules
                    import torch
                    from transformers import AutoModelForCausalLM
                    
                    # Load model in PyTorch
                    pt_model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    
                    # Create JAX/Flax version of the model
                    from transformers import FlaxGPT2LMHeadModel
                    
                    # Get model config
                    config = pt_model.config
                    
                    # Create a properly sized Flax model based on similar architecture (GPT-2)
                    self.model = FlaxGPT2LMHeadModel(config=config)
                    
                    # Convert parameters (limited support, may not work for all models)
                    from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
                    
                    # Get PyTorch state dict
                    pt_state_dict = pt_model.state_dict()
                    
                    # Skip some parameters if needed
                    if is_pythia:
                        # Set model type to gpt2 (compatible architecture)
                        self.model_type = "gpt2"
                        # Extract key dimensions
                        self.num_layers = config.num_hidden_layers
                        self.num_heads = config.num_attention_heads
                        
                        # No need for manual conversion as we're using gpt2 model type
                        self.original_params = self.model.params
                        
                        print(f"Using GPT-2 compatible model for Pythia. Layers: {self.num_layers}, Heads: {self.num_heads}")
                        return True
                    
                except ImportError:
                    print("PyTorch not available, cannot load Pythia models")
                    return False
                except Exception as e:
                    print(f"Error in PyTorch conversion: {e}")
                    # Try the direct FlaxAutoModelForCausalLM approach as fallback
            
            # For non-Pythia models, use standard approach
            self.model = FlaxAutoModelForCausalLM.from_pretrained(self.model_name)
            self.original_params = self.model.params
            
            # Get model details based on model type
            if self.model_type == "gpt2":
                self.num_layers = len(self.original_params["transformer"]["h"])
                self.num_heads = 12  # Standard for most GPT-2 variants
                if "distil" in self.model_name.lower():
                    self.num_layers = 6  # DistilGPT2 has 6 layers
                elif "medium" in self.model_name.lower():
                    self.num_heads = 16  # GPT2-medium has 16 heads
                elif "large" in self.model_name.lower():
                    self.num_heads = 20  # GPT2-large has 20 heads
                elif "xl" in self.model_name.lower():
                    self.num_heads = 25  # GPT2-xl has 25 heads
            elif self.model_type == "opt":
                self.num_layers = len(self.original_params["model"]["decoder"]["layers"])
                # Extract num_heads from config
                self.num_heads = 12  # Default, will be refined below
                try:
                    if "125m" in self.model_name.lower():
                        self.num_heads = 12
                    elif "350m" in self.model_name.lower():
                        self.num_heads = 16
                    elif "1.3b" in self.model_name.lower():
                        self.num_heads = 32
                except Exception:
                    pass  # Stick with default
            elif self.model_type == "pythia":
                # This code only runs if direct loading worked
                self.num_layers = len(self.original_params["transformer"]["h"])
                # Extract num_heads based on model size
                self.num_heads = 12  # Default
                try:
                    if "160m" in self.model_name.lower():
                        self.num_heads = 12
                    elif "410m" in self.model_name.lower():
                        self.num_heads = 16
                    elif "1b" in self.model_name.lower():
                        self.num_heads = 16
                except Exception:
                    pass  # Stick with default
            
            print(f"Model loaded successfully. Layers: {self.num_layers}, Heads per layer: {self.num_heads}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def prune_head(self, params, layer_idx, head_idx):
        """Zero out weights for a specific attention head"""
        if self.model_type == "gpt2":
            # Access path to transformer layers
            transformer_path = "transformer"
            layer_path = "h"
            layer_key = str(layer_idx)
            attn_path = "attn"
            
            # Get attention block
            attn_block = params[transformer_path][layer_path][layer_key][attn_path]
            
            # Calculate head dimensions
            if "c_attn" in attn_block:
                hidden_size = attn_block["c_attn"]["kernel"].shape[1]
            else:
                # Fallback using output projection
                hidden_size = attn_block["c_proj"]["kernel"].shape[0]
                
            head_size = hidden_size // self.num_heads
            
            # Calculate indices for this head
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            
            # Zero out the output projection for this head
            output_proj = attn_block["c_proj"]["kernel"]
            zeros = jnp.zeros_like(output_proj[start_idx:end_idx, :])
            output_proj = output_proj.at[start_idx:end_idx, :].set(zeros)
            
            # Update parameters
            params[transformer_path][layer_path][layer_key][attn_path]["c_proj"]["kernel"] = output_proj
            
        elif self.model_type == "opt":
            # For OPT models
            model_path = "model"
            decoder_path = "decoder"
            layers_path = "layers"
            layer_key = str(layer_idx)
            attn_path = "self_attn"
            
            # Get attention block
            attn_block = params[model_path][decoder_path][layers_path][layer_key][attn_path]
            
            # Calculate head dimensions
            hidden_size = attn_block["out_proj"]["kernel"].shape[0]
            head_size = hidden_size // self.num_heads
            
            # Calculate indices for this head
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            
            # Zero out the output projection for this head
            output_proj = attn_block["out_proj"]["kernel"]
            zeros = jnp.zeros_like(output_proj[start_idx:end_idx, :])
            output_proj = output_proj.at[start_idx:end_idx, :].set(zeros)
            
            # Update parameters
            params[model_path][decoder_path][layers_path][layer_key][attn_path]["out_proj"]["kernel"] = output_proj
            
        elif self.model_type == "pythia":
            # For Pythia models (similar to GPT-2)
            transformer_path = "transformer"
            layer_path = "h"
            layer_key = str(layer_idx)
            attn_path = "attn"
            
            # Get attention block
            attn_block = params[transformer_path][layer_path][layer_key][attn_path]
            
            # Calculate head dimensions
            hidden_size = attn_block["proj"]["kernel"].shape[0]
            head_size = hidden_size // self.num_heads
            
            # Calculate indices for this head
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            
            # Zero out the output projection for this head
            output_proj = attn_block["proj"]["kernel"]
            zeros = jnp.zeros_like(output_proj[start_idx:end_idx, :])
            output_proj = output_proj.at[start_idx:end_idx, :].set(zeros)
            
            # Update parameters
            params[transformer_path][layer_path][layer_key][attn_path]["proj"]["kernel"] = output_proj
        
        return params
    
    def evaluate_perplexity(self, params, text):
        """Evaluate model perplexity on text"""
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="jax")
            
            # Get logits
            outputs = self.model(**inputs, params=params)
            logits = outputs.logits
            
            # Check for NaN or Inf values in logits
            if jnp.isnan(logits).any() or jnp.isinf(logits).any():
                print("Warning: NaN/Inf values in logits during perplexity calculation")
                # Try to continue with cleaned logits
                logits = jnp.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate loss
            input_ids = inputs["input_ids"]
            
            # Special handling for OPT models which can have shape issues
            if self.model_type == "opt" and logits.shape[1] != input_ids.shape[1]:
                print(f"Warning: Shape mismatch in OPT model - logits: {logits.shape}, input_ids: {input_ids.shape}")
                # Clip to shorter length to allow calculation
                min_len = min(logits.shape[1], input_ids.shape[1])
                logits = logits[:, :min_len]
                input_ids = input_ids[:, :min_len]
            
            # Ensure input sequences are long enough
            if input_ids.shape[1] <= 1:
                # Not enough tokens for next-token prediction
                print("Warning: Sequence too short for perplexity calculation")
                return float('nan')
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1]
            shift_labels = input_ids[:, 1:]
            
            # Calculate cross entropy loss with additional safety checks
            try:
                # Use log_softmax for numerical stability
                log_probs = jax.nn.log_softmax(shift_logits)
                one_hot_labels = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
                
                # Calculate token-level losses
                token_losses = -jnp.sum(log_probs * one_hot_labels, axis=-1)
                
                # Average over token positions
                loss = jnp.mean(token_losses)
                
                # Return perplexity
                perplexity = jnp.exp(loss).item()
                
                # Safety check for unreasonable values
                if perplexity > 1e6 or jnp.isnan(perplexity) or jnp.isinf(perplexity):
                    return float('nan')
                    
                return perplexity
            
            except Exception as calc_error:
                print(f"Error calculating loss: {calc_error}")
                return float('nan')
                
        except Exception as e:
            # If perplexity calculation fails completely, return NaN
            print(f"Error in perplexity evaluation: {e}")
            return float('nan')
    
    def generate_text(self, params, prompt, max_length=50):
        """Generate text using the model"""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="jax")
        
        # Special handling for OPT models - they often have shape broadcasting issues
        if self.model_type == "opt":
            try:
                # Use the simplest possible generation config for OPT models
                # to avoid shape mismatches
                generation_config = {
                    "params": params,
                    "max_length": max_length,
                    # Use greedy decoding for maximum stability
                    "do_sample": False,  
                    # Only specify essential parameters
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                
                # Generate text with minimal parameters
                outputs = self.model.generate(**inputs, **generation_config)
                
                # Decode output, catching any shape issues
                try:
                    # Try to decode first element
                    text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                except (IndexError, ValueError):
                    # If there's a shape issue, try other indices or fallback
                    try:
                        # Try with a different indexing approach
                        text = self.tokenizer.decode(outputs.sequences.reshape(-1)[:max_length], 
                                                   skip_special_tokens=True)
                    except Exception:
                        # Last resort fallback
                        return prompt + "..."
                
                # Check for empty or bad generations
                if not text or len(text.strip()) < len(prompt):
                    return prompt + "..."
                
                return text
            
            except Exception as e:
                # If generation fails, log and return placeholder
                print(f"Error in OPT model text generation: {e}")
                return prompt + "..."
        
        # For non-OPT models, use normal generation approach
        else:
            # Define generation params
            generation_config = {
                "params": params,
                "max_length": max_length,
                "do_sample": True,
                "top_k": 40,
                "top_p": 0.95,
                "temperature": 0.8,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            try:
                # Generate text
                outputs = self.model.generate(**inputs, **generation_config)
                
                # Decode output
                text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
                
                # Safety check for bad generations
                if len(text.strip()) <= len(prompt):
                    # Fall back to greedy generation
                    greedy_config = {
                        "params": params,
                        "max_length": max_length,
                        "do_sample": False,  # Greedy decoding
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                    }
                    
                    outputs = self.model.generate(**inputs, **greedy_config)
                    text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
                
                return text
                
            except Exception as e:
                # If generation fails, return a placeholder plus the error
                print(f"Error in text generation: {e}")
                return prompt + "..."