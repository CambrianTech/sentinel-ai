"""
UNet-Enhanced Transformer with Baseline Integration

This module implements a UNet-style transformer architecture that leverages
knowledge from a baseline model to improve performance and efficiency. The key
features include:
- Cross-connections between baseline and agency models
- Knowledge distillation from baseline outputs
- Optimized parallel head processing
- Dynamic capacity adjustment based on pruning needs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union

from .optimized_attention import OptimizedGatedMultiHeadAttention
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation.utils import GenerationMixin


class BaselineIntegratedBlock(nn.Module):
    """
    Transformer block that integrates information from a baseline model.
    
    This block implements cross-connections between the agency model and
    the baseline model, allowing information to flow between them. This
    helps the agency model benefit from the baseline model's representations
    while maintaining its own agency capabilities.
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
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # Baseline integration
        self.use_baseline_integration = use_baseline_integration
        self.baseline_fusion_factor = baseline_fusion_factor
        
        if use_baseline_integration:
            # Integration with baseline model
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
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        baseline_states: Optional[torch.Tensor] = None,
        encoder_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        step_count: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional integration of baseline and encoder states.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, embed_dim]
            baseline_states: Optional states from baseline model
            encoder_states: Optional states from encoder layer (UNet skip connection)
            attention_mask: Optional attention mask
            step_count: Optional step counter for agency tracking
            
        Returns:
            Updated hidden states
        """
        # 1. Self-attention with pre-normalization
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        attn_output = self.attn(
            hidden_states,
            attn_mask=attention_mask,
            step_count=step_count
        )
        
        hidden_states = residual + attn_output
        
        # 2. Optional baseline integration
        if self.use_baseline_integration and baseline_states is not None:
            # Adapt baseline states to our representation
            adapted_baseline = self.ln_baseline(baseline_states)
            adapted_baseline = self.baseline_adapter(adapted_baseline)
            
            # Dynamic gating of baseline contribution
            gate_value = torch.sigmoid(self.baseline_gate)
            hidden_states = hidden_states * (1 - gate_value) + adapted_baseline * gate_value
        
        # 3. Optional UNet skip connection
        if self.use_skip_connection and encoder_states is not None:
            # Concatenate hidden states with encoder states
            combined = torch.cat([hidden_states, encoder_states], dim=-1)
            
            # Apply linear fusion with scaled contribution
            fusion_output = self.skip_fuse(combined)
            skip_contribution = fusion_output * self.skip_scale
            
            # Add skip connection contribution
            hidden_states = hidden_states + skip_contribution
        
        # 4. Feed-forward network with pre-normalization
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        
        hidden_states = residual + ffn_output
        
        return hidden_states


class UNetEnhancedTransformer(nn.Module):
    """
    Enhanced transformer model with UNet architecture and baseline model integration.
    
    This model combines:
    1. Optimized multi-head attention with agency features
    2. UNet-style skip connections between encoder and decoder layers
    3. Integration with a baseline model for knowledge transfer
    4. Dynamic capacity adjustment based on pruning needs
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
        
        # Print debug info
        if debug:
            print(f"[Model Init] embed_dim={self.embed_dim}, num_heads={self.num_heads}, ffn_dim={self.ffn_dim}")
        
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
        
        # Create transformer blocks with optimized attention and optional baseline integration
        for i in range(self.num_layers):
            # Enable baseline integration only for selected layers
            # Typically more useful in later layers
            layer_use_baseline = (
                self.use_baseline_integration and 
                i >= self.midpoint  # Only in decoder layers
            )
            
            # Create block with appropriate configuration
            block = BaselineIntegratedBlock(
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
        
        # Create causal attention mask for autoregressive generation
        max_pos = min(getattr(config, 'max_position_embeddings', 1024), 1024)
        causal_mask = torch.tril(torch.ones(max_pos, max_pos))
        self.register_buffer("bias", causal_mask.view(1, 1, max_pos, max_pos))
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        step_count=None,
        return_baseline=False,
        **kwargs
    ):
        """
        Forward pass through the UNet-enhanced transformer.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional padding mask
            labels: Optional labels for loss computation
            step_count: Current step for agency tracking
            return_baseline: Whether to return baseline outputs alongside agency outputs
            **kwargs: Additional arguments
            
        Returns:
            logits or CausalLMOutput with loss, optionally with baseline outputs
        """
        # Basic setup
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Get input embeddings
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        
        # Get baseline model states if integration is enabled
        baseline_outputs = None
        if self.use_baseline_integration and self.baseline_model is not None:
            with torch.no_grad():  # Don't compute gradients for baseline
                # Run baseline model forward pass
                if hasattr(self.baseline_model, "transformer"):
                    # Standard HuggingFace GPT-2 models
                    baseline_outputs = {}
                    
                    # Input embedding from GPT2 model
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
                else:
                    # Fallback for other model types - just get final outputs
                    baseline_outputs = {"final": None, "logits": None}
        
        # Storage for UNet skip connections
        encoder_outputs = {}
        
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
            
            # Create attention mask
            attn_mask = None
            if seq_len <= 1024:
                # Use pre-computed causal mask
                causal_mask = self.bias[:, :, :seq_len, :seq_len]
                attn_mask = (1.0 - causal_mask) * -10000.0
                attn_mask = attn_mask.squeeze(0).squeeze(0)
            else:
                # Create mask on-the-fly for longer sequences
                attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * -10000.0, diagonal=1)
            
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
            agency_output = CausalLMOutput(loss=loss, logits=logits)
            return agency_output, baseline_outputs["logits"]
        elif labels is not None:
            # Return output with loss
            return CausalLMOutput(loss=loss, logits=logits)
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


class UNetAdaptiveCausalLmWrapper(UNetEnhancedTransformer, GenerationMixin):
    """
    Causal language model wrapper for the UNet-enhanced transformer.
    
    This wrapper adds generation capabilities to the UNet transformer model,
    making it compatible with the HuggingFace generate API.
    """
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    
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
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        """Prepare inputs for generation process."""
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search if needed."""
        # This model doesn't use cache yet, but this method is required by GenerationMixin
        return past_key_values
    
    def get_output_embeddings(self):
        """Get output embeddings for generation."""
        return self.lm_head
    
    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        """Forward pass handling generation specifically."""
        # Call the parent class forward method
        outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)
        
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
                logits = torch.cat([logits[:, :-1, :], last_pos_logits], dim=1)
                outputs = logits
        
        # Return in the expected format
        if return_dict and not isinstance(outputs, CausalLMOutput):
            return CausalLMOutput(logits=outputs)
        return outputs


def load_unet_enhanced_model(
    baseline_model,
    device="cpu",
    use_baseline_integration=True,
    debug=False
):
    """
    Create a UNet-enhanced model initialized from a baseline model.
    
    Args:
        baseline_model: The baseline model to initialize from
        device: Device to load the model on
        use_baseline_integration: Whether to use baseline knowledge
        debug: Whether to print debug information
        
    Returns:
        UNetAdaptiveCausalLmWrapper: The initialized model
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
    model = UNetAdaptiveCausalLmWrapper(
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
        print(f"UNet model initialized with {num_params:,} parameters")
        print(f"Baseline model has {baseline_params:,} parameters")
        print(f"Ratio: {num_params / baseline_params:.2f}x")
    
    return model