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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
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
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="jax")
        
        # Get logits
        outputs = self.model(**inputs, params=params)
        logits = outputs.logits
        
        # Calculate loss
        input_ids = inputs["input_ids"]
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[:, :-1]
        shift_labels = input_ids[:, 1:]
        
        # Calculate cross entropy loss
        loss = jnp.mean(
            -jnp.sum(
                jax.nn.log_softmax(shift_logits) * jax.nn.one_hot(shift_labels, shift_logits.shape[-1]),
                axis=-1
            )
        )
        
        # Return perplexity
        return jnp.exp(loss).item()
    
    def generate_text(self, params, prompt, max_length=50):
        """Generate text using the model"""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="jax")
        
        # Define generation params based on model type
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
        
        # Special handling for OPT models which can have generation issues after pruning
        if self.model_type == "opt":
            # Use more conservative sampling settings
            generation_config.update({
                "temperature": 0.7,  # Lower temperature for more predictable outputs
                "top_p": 0.85,       # More restrictive nucleus sampling
                "top_k": 20,         # More restrictive top-k sampling
                "no_repeat_ngram_size": 2,  # Avoid repeating bigrams
                "min_length": len(inputs.input_ids[0]) + 1,  # Ensure generation happens
                "early_stopping": True,  # Stop when model would generate EOS
            })
        
        # Generate text
        try:
            outputs = self.model.generate(**inputs, **generation_config)
            
            # Decode output
            text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            
            # Safety check for bad generations (especially with OPT models)
            if len(text.strip()) <= len(prompt) or "<s>" in text or "</s>" in text:
                # Fall back to greedy generation
                outputs = self.model.generate(
                    **inputs,
                    params=params,
                    max_length=max_length,
                    do_sample=False,  # Greedy decoding
                    num_beams=1,      # No beam search
                )
                text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            
            return text
        except Exception as e:
            # If generation fails, return a placeholder plus the error
            print(f"Error in text generation: {e}")
            return prompt + "..."