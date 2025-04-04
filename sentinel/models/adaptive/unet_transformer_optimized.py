"""
Optimized UNet Transformer with Efficient Baseline Integration

This module provides an optimized implementation of the UNet-style transformer
with more efficient baseline model integration. It addresses performance bottlenecks
identified in the original implementation by:

1. Minimizing redundant tensor operations
2. Reducing CPU-GPU synchronization points  
3. Optimizing knowledge transfer from the baseline model
4. Caching intermediate results where appropriate
5. Using in-place operations where possible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union

from sentinel.models.utils.optimized_attention import OptimizedGatedMultiHeadAttention
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation.utils import GenerationMixin


class OptimizedBaselineIntegratedBlock(nn.Module):
    """
    Optimized transformer block with efficient baseline model integration.
    
    Improvements over the original implementation:
    - Minimizes redundant computation and tensor operations
    - Uses in-place operations where possible
    - Caches intermediate results to avoid recomputation
    - Optimizes the knowledge transfer mechanism
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int = None,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        activation: str = "gelu",
        use_baseline_integration: bool = True,
        baseline_fusion_factor: float = 0.3,
    ):
        super().__init__()
        
        # Default FFN dimension to 4x embedding dimension if not specified
        if ffn_dim is None:
            ffn_dim = 4 * embed_dim
        
        # Optimized multi-head attention with agency features
        self.attn = OptimizedGatedMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network with fused activation
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization - use a single instance for parallel computation
        self.ln1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # Baseline integration
        self.use_baseline_integration = use_baseline_integration
        self.baseline_fusion_factor = baseline_fusion_factor
        
        if use_baseline_integration:
            # Integration with baseline model (more efficient implementation)
            self.baseline_adapter = nn.Linear(embed_dim, embed_dim)
            self.baseline_gate = nn.Parameter(torch.ones(1) * baseline_fusion_factor)
            self.ln_baseline = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
            
            # Skip connection from encoder to decoder (UNet style)
            self.skip_fuse = nn.Linear(2 * embed_dim, embed_dim)
            self.skip_gate = nn.Parameter(torch.ones(1) * 0.1)  # Start with small value
            
            # Configuration attributes for skip connections
            self.use_skip_connection = False
            self.skip_source = -1
            self.skip_scale = 0.1
            
            # Caching for intermediate computation
            self.register_buffer("_baseline_cache", None, persistent=False)
            self.register_buffer("_encoder_cache", None, persistent=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        baseline_states: Optional[torch.Tensor] = None,
        encoder_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        step_count: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Optimized forward pass with efficient baseline model integration.
        """
        # 1. Self-attention with pre-normalization - use direct tensor operations
        residual = hidden_states
        
        # Apply layer norm once
        norm_hidden = self.ln1(hidden_states)
        
        # Apply attention
        attn_output = self.attn(
            norm_hidden,
            attn_mask=attention_mask,
            step_count=step_count
        )
        
        # Residual connection (in-place where possible)
        if hidden_states.requires_grad:
            hidden_states = residual + attn_output
        else:
            hidden_states.copy_(residual + attn_output)
        
        # 2. Baseline model integration - fuse operations where possible
        if self.use_baseline_integration and baseline_states is not None:
            # Use cached baseline transform if possible
            if hasattr(baseline_states, "_baseline_transformed"):
                adapted_baseline = baseline_states._baseline_transformed
            else:
                # Apply normalization and adaptation in a single pass
                adapted_baseline = self.baseline_adapter(self.ln_baseline(baseline_states))
                
                # Cache the result if not requiring gradients
                if not baseline_states.requires_grad:
                    baseline_states._baseline_transformed = adapted_baseline
            
            # Apply dynamic gating with efficient broadcast
            gate_value = torch.sigmoid(self.baseline_gate)
            hidden_states = hidden_states * (1 - gate_value) + adapted_baseline * gate_value
        
        # 3. UNet skip connection - only compute when needed
        if self.use_skip_connection and encoder_states is not None:
            # Concatenate once and apply linear fusion
            combined = torch.cat([hidden_states, encoder_states], dim=-1)
            fusion_output = self.skip_fuse(combined)
            
            # Apply scaled contribution in-place where possible
            skip_contribution = fusion_output * self.skip_scale
            if hidden_states.requires_grad:
                hidden_states = hidden_states + skip_contribution
            else:
                hidden_states.add_(skip_contribution)
        
        # 4. Feed-forward network with pre-normalization
        residual = hidden_states
        
        # Apply layer norm once
        norm_hidden = self.ln2(hidden_states)
        
        # Apply FFN
        ffn_output = self.ffn(norm_hidden)
        
        # Residual connection (in-place where possible)
        if hidden_states.requires_grad:
            hidden_states = residual + ffn_output
        else:
            hidden_states.copy_(residual + ffn_output)
        
        return hidden_states


class OptimizedUNetTransformer(nn.Module):
    """
    Optimized UNet-style transformer with efficient baseline integration.
    
    This model improves performance over the original implementation by:
    - Using optimized attention mechanism
    - Minimizing redundant computation
    - Optimizing data flow between components
    - Caching intermediate results when appropriate
    - Reducing CPU-GPU synchronization points
    """
    def __init__(
        self,
        config,
        token_embeddings,
        position_embeddings,
        baseline_model=None,
        use_baseline_integration=True,
        debug=False
    ):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size if hasattr(config, 'hidden_size') else config.n_embd
        self.num_heads = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_head
        self.num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layer
        self.debug = debug
        self.enable_agency = True
        
        # Determine FFN dimension from config
        self.ffn_dim = (
            getattr(config, 'n_inner', None) or 
            getattr(config, 'intermediate_size', None) or 
            4 * self.embed_dim
        )
        
        # Use provided embeddings
        self.wte = token_embeddings
        self.wpe = position_embeddings
        
        # Store baseline model for knowledge transfer
        self.baseline_model = baseline_model
        self.use_baseline_integration = use_baseline_integration and baseline_model is not None
        
        # Initialize transformer blocks
        self.blocks = nn.ModuleList()
        
        # Define midpoint for UNet architecture
        self.midpoint = self.num_layers // 2
        
        # Create transformer blocks with optimized implementation
        for i in range(self.num_layers):
            # Enable baseline integration only for selected layers
            # Typically more useful in later layers
            layer_use_baseline = (
                self.use_baseline_integration and 
                i >= self.midpoint  # Only in decoder layers
            )
            
            # Create block with optimized implementation
            block = OptimizedBaselineIntegratedBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                use_baseline_integration=layer_use_baseline,
                baseline_fusion_factor=0.3 if layer_use_baseline else 0.0
            )
            
            # Configure UNet skip connections
            if i >= self.midpoint:  # Decoder layers
                # Connect to corresponding encoder layer
                encoder_idx = self.num_layers - i - 1
                if encoder_idx >= 0:
                    block.use_skip_connection = True
                    block.skip_source = encoder_idx
                    # Gradually increase skip connection strength in deeper layers
                    block.skip_scale = 0.1 * (1 + (i - self.midpoint) / (self.num_layers - self.midpoint))
            
            self.blocks.append(block)
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, config.vocab_size, bias=False)
        
        # Tie weights between input embeddings and output layer
        self.lm_head.weight = self.wte.weight
        
        # Create causal attention mask only once and cache it
        max_pos = min(getattr(config, 'max_position_embeddings', 1024), 1024)
        causal_mask = torch.tril(torch.ones(max_pos, max_pos))
        self.register_buffer("bias", causal_mask.view(1, 1, max_pos, max_pos))
        
        # Cache for baseline model outputs - improves performance during generation
        self._baseline_cache = None
    
    def _run_baseline_model(self, input_ids, attention_mask=None, past_key_values=None):
        """Run baseline model and cache results efficiently."""
        # Return cached results if available
        if self._baseline_cache is not None and past_key_values is not None:
            return self._baseline_cache
        
        baseline_outputs = {}
        
        with torch.no_grad():  # Don't compute gradients for baseline
            # Get baseline model depending on structure
            if hasattr(self.baseline_model, "transformer"):
                # Input embedding from GPT2 model
                batch_size, seq_len = input_ids.shape
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                
                baseline_embeds = self.baseline_model.transformer.wte(input_ids)
                baseline_pos = self.baseline_model.transformer.wpe(position_ids)
                baseline_h = baseline_embeds + baseline_pos
                
                # Process through each transformer block
                for i, block in enumerate(self.baseline_model.transformer.h):
                    baseline_h = block(baseline_h)[0]  # GPT2 blocks return a tuple
                    baseline_outputs[i] = baseline_h  # Store each layer's output
                
                # Final normalization
                baseline_h = self.baseline_model.transformer.ln_f(baseline_h)
                baseline_logits = self.baseline_model.lm_head(baseline_h)
                baseline_outputs["final"] = baseline_h
                baseline_outputs["logits"] = baseline_logits
        
        # Cache results for generation if using past_key_values
        if past_key_values is not None:
            self._baseline_cache = baseline_outputs
        
        return baseline_outputs
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        step_count=None,
        return_baseline=False,
        past_key_values=None,
        **kwargs
    ):
        """Optimized forward pass through the UNet transformer."""
        # Basic setup
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Clear cache if not in generation mode (new prompt)
        if past_key_values is None:
            self._baseline_cache = None
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Get input embeddings
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        
        # Get baseline model outputs if integration is enabled
        baseline_outputs = None
        if self.use_baseline_integration and self.baseline_model is not None:
            baseline_outputs = self._run_baseline_model(input_ids, attention_mask, past_key_values)
        
        # Storage for UNet skip connections
        encoder_outputs = {}
        
        # Create attention mask once - optimized for generation
        if seq_len <= 1024:
            # Use pre-computed causal mask
            causal_mask = self.bias[:, :, :seq_len, :seq_len]
            attn_mask = (1.0 - causal_mask) * -10000.0
            attn_mask = attn_mask.squeeze(0).squeeze(0)
        else:
            # Create mask on-the-fly for longer sequences
            attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * -10000.0, diagonal=1)
        
        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            # In encoder layers, store outputs for later UNet connections
            if i < self.midpoint:
                encoder_outputs[i] = hidden_states.clone()
            
            # Get baseline hidden states for this layer if available
            baseline_states = None
            if baseline_outputs is not None and i in baseline_outputs:
                baseline_states = baseline_outputs[i]
            
            # Get encoder states for UNet skip connection if this is a decoder layer
            encoder_states = None
            if block.use_skip_connection and block.skip_source in encoder_outputs:
                encoder_states = encoder_outputs[block.skip_source]
            
            # Process through the block
            hidden_states = block(
                hidden_states,
                baseline_states=baseline_states,
                encoder_states=encoder_states,
                attention_mask=attn_mask,
                step_count=step_count
            )
        
        # Final layer normalization
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits through language model head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift for autoregressive loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Return appropriate output format
        if return_baseline and baseline_outputs is not None:
            # Return both agency and baseline outputs
            agency_output = CausalLMOutput(loss=loss, logits=logits, past_key_values=past_key_values)
            return agency_output, baseline_outputs["logits"]
        elif labels is not None:
            # Return output with loss
            return CausalLMOutput(loss=loss, logits=logits, past_key_values=past_key_values)
        else:
            # Return logits only
            return logits
    
    def get_gate_activity(self):
        """Returns a dictionary with gate activity information for analysis."""
        gate_activity = {}
        for layer_idx, block in enumerate(self.blocks):
            # Get gate values from attention module
            if hasattr(block.attn, "gate"):
                gate_values = block.attn.gate.detach().cpu()
                # Get indices of active heads (gate value > threshold)
                active_heads = [i for i, g in enumerate(gate_values) if float(g) > 0.2]
                gate_activity[layer_idx] = active_heads
        return gate_activity
    
    def get_agency_report(self):
        """Generate a comprehensive report on agency status across all layers."""
        layer_reports = {}
        total_violations = 0
        
        for i, block in enumerate(self.blocks):
            if hasattr(block.attn, "get_agency_report"):
                report = block.attn.get_agency_report()
                layer_reports[i] = report
                total_violations += report["violation_count"]
        
        # Add UNet connection information
        unet_connections = {}
        for i, block in enumerate(self.blocks):
            if hasattr(block, "use_skip_connection") and block.use_skip_connection:
                unet_connections[i] = {
                    "source": block.skip_source,
                    "scale": float(block.skip_scale)
                }
        
        # Add baseline integration information
        baseline_integration = {}
        for i, block in enumerate(self.blocks):
            if hasattr(block, "use_baseline_integration") and block.use_baseline_integration:
                if hasattr(block, "baseline_gate"):
                    gate_value = float(torch.sigmoid(block.baseline_gate))
                    baseline_integration[i] = gate_value
        
        return {
            "layer_reports": layer_reports,
            "unet_connections": unet_connections,
            "baseline_integration": baseline_integration,
            "total_violations": total_violations,
            "num_layers": self.num_layers,
        }
    
    def set_head_state(self, layer_idx, head_idx, state, consent=None):
        """Set state and consent for a specific attention head."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return False
            
        block = self.blocks[layer_idx]
        if hasattr(block.attn, "set_head_state"):
            return block.attn.set_head_state(head_idx, state, consent)
        return False


class OptimizedUNetCausalLmWrapper(OptimizedUNetTransformer, GenerationMixin):
    """
    Optimized causal language model wrapper for the UNet transformer.
    
    This wrapper optimizes the generation process to make better use of caching
    and minimize redundant computation during text generation.
    """
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    can_generate = True  # Required for newer transformers versions
    _supports_cache_class = False  # Required for newer transformers versions
    
    def __init__(
        self,
        config,
        token_embeddings,
        position_embeddings,
        baseline_model=None,
        use_baseline_integration=True,
        debug=False
    ):
        super().__init__(
            config,
            token_embeddings,
            position_embeddings,
            baseline_model,
            use_baseline_integration,
            debug
        )
        
        # Set up generation configuration
        from transformers import GenerationConfig
        try:
            self.generation_config = GenerationConfig.from_model_config(config)
        except Exception as e:
            if debug:
                print(f"Warning: Could not create generation config, using defaults: {e}")
            self.generation_config = GenerationConfig()
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        """Prepare inputs for generation process with efficient caching."""
        # For the first generation step, run normally
        if past_key_values is None:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": None
            }
        
        # For subsequent steps, only need the new token
        input_ids = input_ids[:, -1].unsqueeze(-1)
        
        # Build generation inputs
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values
        }
    
    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search if needed."""
        # This model doesn't use cache yet, but this method is required by GenerationMixin
        if self._baseline_cache is not None:
            # Reorder the baseline cache
            reordered_cache = {}
            for k, v in self._baseline_cache.items():
                if isinstance(v, torch.Tensor):
                    reordered_cache[k] = v.index_select(0, beam_idx)
                else:
                    reordered_cache[k] = v
            self._baseline_cache = reordered_cache
        
        return past_key_values
        
    def can_generate(self):
        """Method to check if the model can generate text."""
        return True
        
    @property
    def device(self):
        """Return device of the first parameter of the model."""
        return next(self.parameters()).device
    
    def get_output_embeddings(self):
        """Get output embeddings for generation."""
        return self.lm_head
    
    def forward(self, input_ids, attention_mask=None, return_dict=True, past_key_values=None, **kwargs):
        """Forward pass optimized for generation efficiency."""
        # Call the parent class forward method
        outputs = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs
        )
        
        # If we're in generation mode, apply post-processing
        if isinstance(outputs, torch.Tensor) and attention_mask is not None:
            # Extract batch size and sequence length
            batch_size, seq_len = input_ids.shape
            
            # Check if we're in generation mode (typically sequence length > 1)
            is_generation = seq_len > 1
            
            if is_generation:
                # For generation, apply mild temperature scaling to improve output quality
                logits = outputs
                
                # Only modify logits for the last position in generation mode
                last_pos_logits = logits[:, -1:, :]
                
                # Apply mild temperature for better distribution
                temperature = 1.0
                last_pos_logits = last_pos_logits / temperature
                
                # Combine them back
                if seq_len > 1:
                    logits = torch.cat([logits[:, :-1, :], last_pos_logits], dim=1)
                else:
                    logits = last_pos_logits
                
                outputs = logits
        
        # Return in the expected format
        if return_dict and not isinstance(outputs, CausalLMOutput):
            return CausalLMOutput(logits=outputs, past_key_values=past_key_values)
        return outputs


def load_optimized_unet_model(
    baseline_model,
    device="cpu",
    use_baseline_integration=True,
    debug=False
):
    """
    Create an optimized UNet model initialized from a baseline model.
    
    Args:
        baseline_model: The baseline model to initialize from
        device: Device to load the model on
        use_baseline_integration: Whether to use baseline knowledge
        debug: Whether to print debug information
        
    Returns:
        OptimizedUNetCausalLmWrapper: The initialized model
    """
    config = baseline_model.config
    
    # Get token and position embeddings from baseline model
    if hasattr(baseline_model, "transformer"):
        # Standard HuggingFace models (GPT-2)
        token_embeddings = baseline_model.transformer.wte
        position_embeddings = baseline_model.transformer.wpe
    else:
        # Try generic approach
        print("Warning: Non-standard model structure. Embeddings might not be correctly initialized.")
        token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    
    # Create and initialize the model
    model = OptimizedUNetCausalLmWrapper(
        config=config,
        token_embeddings=token_embeddings,
        position_embeddings=position_embeddings,
        baseline_model=baseline_model if use_baseline_integration else None,
        use_baseline_integration=use_baseline_integration,
        debug=debug
    )
    
    # Move to device
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    if debug:
        num_params = sum(p.numel() for p in model.parameters())
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        print(f"Optimized UNet model initialized with {num_params:,} parameters")
        print(f"Baseline model has {baseline_params:,} parameters")
        print(f"Ratio: {num_params / baseline_params:.2f}x")
    
    return model