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

from sentinel.models.utils.optimized_attention import OptimizedGatedMultiHeadAttention
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation.utils import GenerationMixin


class BaselineIntegratedBlock(nn.Module):
    """
    Transformer block that integrates information from a baseline model.
    
    This block implements cross-connections between the agency model and
    the baseline model, allowing information to flow between them. This
    helps the agency model benefit from the baseline model's representations
    while maintaining the ability to prune and adapt.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int = None,
        dropout_prob: float = 0.1,
        activation: str = "gelu",
        baseline_connection_weight: float = 0.1,
        layer_norm_eps: float = 1e-12,
        debug: bool = False
    ):
        super().__init__()
        
        # Default intermediate size to 4x hidden size
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
            
        # Core transformer components
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = OptimizedGatedMultiHeadAttention(
            hidden_size, 
            num_heads,
            dropout_prob=dropout_prob,
            debug=debug
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout_prob)
        )
        
        # Baseline integration components
        self.baseline_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.baseline_projection = nn.Linear(hidden_size, hidden_size)
        self.baseline_gate = nn.Parameter(torch.ones(1) * baseline_connection_weight)
        
        # U-Net skip connection components
        self.use_skip_connection = False
        self.skip_source = None
        self.skip_scale = 0.1
        
        # For tracking attention weights
        self.store_attention_weights = False
        
    def forward(
        self,
        x: torch.Tensor,
        baseline_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        skip_memory: Optional[Dict[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass integrating baseline model knowledge.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            baseline_hidden_states: Hidden states from baseline model
            attention_mask: Attention mask [batch_size, seq_len]
            skip_memory: Dictionary of layer outputs for skip connections
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Save input for residual
        residual = x
        
        # Set flag for storing attention weights if needed
        if hasattr(self.attn, 'store_attention_weights'):
            self.attn.store_attention_weights = self.store_attention_weights
        
        # Attention branch
        hidden_states = self.ln_1(x)
        attn_output = self.attn(hidden_states, attention_mask=attention_mask)
        
        # Add residual connection
        hidden_states = residual + attn_output
        
        # Apply baseline integration if provided
        if baseline_hidden_states is not None:
            # Process baseline hidden states
            baseline_proj = self.baseline_projection(
                self.baseline_ln(baseline_hidden_states)
            )
            
            # Apply gated connection
            hidden_states = hidden_states + self.baseline_gate * baseline_proj
            
        # Apply U-Net skip connection if enabled
        if self.use_skip_connection and self.skip_source is not None and skip_memory is not None:
            if self.skip_source in skip_memory:
                skip_input = skip_memory[self.skip_source]
                if skip_input.shape == hidden_states.shape:
                    hidden_states = hidden_states + self.skip_scale * skip_input
                    
        # Save for second residual
        residual = hidden_states
        
        # Feed-forward branch
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Add second residual
        output = residual + hidden_states
        
        return output


class UNetTransformer(nn.Module):
    """
    UNet-style transformer integrating a baseline model.
    
    This architecture provides several key benefits:
    1. Efficient pruning through gated attention heads
    2. U-Net style skip connections between encoder and decoder layers
    3. Baseline model integration for knowledge transfer
    4. Agency-aware attention mechanisms
    """
    
    def __init__(
        self,
        config,
        baseline_model=None,
        use_baseline_integration=True,
        connection_scale=0.1,
        debug=False
    ):
        super().__init__()
        
        # Save configuration
        self.config = config
        self.debug = debug
        
        # Extract dimensions from config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 1024)
        self.num_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_head', 12))
        self.num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 12))
        self.dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)
        self.intermediate_size = getattr(config, 'intermediate_size', 4 * self.hidden_size)
        
        # Embedding layers
        self.wte = nn.Embedding(self.vocab_size, self.hidden_size)
        self.wpe = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        self.drop = nn.Dropout(self.dropout_prob)
        
        # Create transformer blocks
        self.blocks = nn.ModuleList([
            BaselineIntegratedBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                dropout_prob=self.dropout_prob,
                baseline_connection_weight=connection_scale,
                debug=debug
            ) for _ in range(self.num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(self.hidden_size)
        
        # Set up the baseline model integration
        self.baseline_model = baseline_model
        self.use_baseline_integration = use_baseline_integration
        
        # Set up skip connections
        self.setup_skip_connections()
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Copy embeddings from baseline model if available
        if baseline_model is not None:
            self._copy_baseline_embeddings()
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def _copy_baseline_embeddings(self):
        """Copy embeddings from baseline model."""
        if self.baseline_model is not None:
            # Copy token embeddings
            if hasattr(self.baseline_model, 'get_input_embeddings'):
                baseline_wte = self.baseline_model.get_input_embeddings()
                self.wte.weight.data.copy_(baseline_wte.weight.data)
                if self.debug:
                    print("Copied token embeddings from baseline model")
            
            # Try to copy position embeddings
            if hasattr(self.baseline_model, 'transformer') and hasattr(self.baseline_model.transformer, 'wpe'):
                self.wpe.weight.data.copy_(self.baseline_model.transformer.wpe.weight.data)
                if self.debug:
                    print("Copied position embeddings from baseline model")
    
    def setup_skip_connections(self):
        """Set up U-Net style skip connections between layers."""
        midpoint = self.num_layers // 2
        
        # For layers in second half (decoder), connect to corresponding layer in first half
        for i in range(midpoint, self.num_layers):
            decoder_idx = i
            encoder_idx = self.num_layers - i - 1
            
            # Only connect if encoder idx is valid
            if encoder_idx >= 0:
                self.blocks[decoder_idx].use_skip_connection = True
                self.blocks[decoder_idx].skip_source = encoder_idx
                
                # Scale decreases as we go deeper (more distant connections are weaker)
                depth_factor = (i - midpoint) / (self.num_layers - midpoint)
                self.blocks[decoder_idx].skip_scale = 0.1 * (1.0 - depth_factor * 0.5)
    
    def get_input_embeddings(self):
        """Get the model's input embeddings."""
        return self.wte
        
    def set_input_embeddings(self, embeddings):
        """Set the model's input embeddings."""
        self.wte = embeddings
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        return_dict=None
    ):
        """
        Forward pass of the UNet Transformer.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            past_key_values: Past key values for efficient generation
            inputs_embeds: Pre-computed input embeddings
            use_cache: Whether to use cache for efficient generation
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            Tuple or dictionary of outputs
        """
        # Default return format
        return_dict = return_dict if return_dict is not None else True
        
        # Get device
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        # Get batch size and sequence length
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]
            
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Create attention mask if not provided but needed
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
            
        # Convert attention mask to float and create extended attention mask
        extended_attention_mask = None
        if attention_mask is not None:
            # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Convert to additive attention mask (0 -> 0, 1 -> -10000)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Process with baseline model if available and integration is enabled
        baseline_outputs = None
        if self.baseline_model is not None and self.use_baseline_integration:
            with torch.no_grad():
                # Get baseline model outputs
                baseline_outputs = self.baseline_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    return_dict=True
                )
        
        # Compute inputs
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        # Add position embeddings
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        # Apply dropout
        hidden_states = self.drop(hidden_states)
        
        # Storage for U-Net skip connections
        layer_outputs = {}
        
        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            # Save layer output for potential skip connections
            layer_outputs[i] = hidden_states
            
            # Get baseline hidden state for this layer if available
            baseline_hidden_state = None
            if baseline_outputs is not None and hasattr(baseline_outputs, "hidden_states"):
                if i < len(baseline_outputs.hidden_states):
                    baseline_hidden_state = baseline_outputs.hidden_states[i]
            
            # Process through block
            hidden_states = block(
                hidden_states,
                baseline_hidden_states=baseline_hidden_state,
                attention_mask=extended_attention_mask,
                skip_memory=layer_outputs
            )
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "hidden_states": layer_outputs
            }
        
        return (hidden_states,)


class UNetLMHeadModel(nn.Module, GenerationMixin):
    """
    UNet transformer model with a language modeling head.
    
    This class integrates with HuggingFace's generation utilities to
    provide a drop-in replacement for standard language models with
    adaptive architecture benefits.
    """
    
    def __init__(
        self,
        config,
        baseline_model=None,
        use_baseline_integration=True,
        connection_scale=0.1,
        debug=False
    ):
        super().__init__()
        
        # Create transformer
        self.transformer = UNetTransformer(
            config=config,
            baseline_model=baseline_model,
            use_baseline_integration=use_baseline_integration,
            connection_scale=connection_scale,
            debug=debug
        )
        
        # Create language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Copy LM head weights from baseline model if available
        if baseline_model is not None and hasattr(baseline_model, "lm_head"):
            self.lm_head.weight.data.copy_(baseline_model.lm_head.weight.data)
            if debug:
                print("Copied LM head weights from baseline model")
        elif baseline_model is not None and hasattr(baseline_model, "get_output_embeddings"):
            output_embeddings = baseline_model.get_output_embeddings()
            if output_embeddings is not None:
                self.lm_head.weight.data.copy_(output_embeddings.weight.data)
                if debug:
                    print("Copied output embeddings from baseline model")
        
        # Store configuration
        self.config = config
        
        # Required attributes for generation
        self.main_input_name = "input_ids"
        
    @property
    def device(self):
        """Get the model's device."""
        return next(self.parameters()).device
        
    def get_output_embeddings(self):
        """Get the model's output embeddings."""
        return self.lm_head
        
    def set_output_embeddings(self, new_embeddings):
        """Set the model's output embeddings."""
        self.lm_head = new_embeddings
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        return_dict=None
    ):
        """
        Forward pass of the language model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            past_key_values: Past key values for efficient generation
            inputs_embeds: Pre-computed input embeddings
            labels: Labels for computing the language modeling loss
            use_cache: Whether to use cache for efficient generation
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            CausalLMOutput with loss, logits, and hidden states
        """
        # Default return format
        return_dict = return_dict if return_dict is not None else True
        
        # Run transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True
        )
        
        # Get last hidden state
        hidden_states = transformer_outputs["last_hidden_state"]
        
        # Apply LM head
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if not return_dict:
            return (loss, logits, transformer_outputs["hidden_states"])
            
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs["hidden_states"],
            attentions=None  # We don't return attention weights by default
        )
    
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        """
        Prepare inputs for generation.
        
        This implements the required method for HuggingFace's GenerationMixin.
        
        Args:
            input_ids: Input token IDs
            past: Past key values for efficient generation
            
        Returns:
            Dictionary of model inputs for generation
        """
        # Only keep last token for inputs_ids if past is provided
        if past is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
        attention_mask = kwargs.get("attention_mask", None)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past
        }


def load_unet_enhanced_model(baseline_model, device="cuda", use_baseline_integration=True, debug=False):
    """
    Load a UNet-enhanced model initialized from a baseline model.
    
    Args:
        baseline_model: The HuggingFace baseline model to enhance
        device: Device to load the model on
        use_baseline_integration: Whether to integrate with baseline model
        debug: Whether to print debug information
        
    Returns:
        Initialized UNetLMHeadModel
    """
    # Get model config
    config = baseline_model.config
    
    # Create UNet model
    unet_model = UNetLMHeadModel(
        config=config,
        baseline_model=baseline_model,
        use_baseline_integration=use_baseline_integration,
        debug=debug
    )
    
    # Move model to device
    unet_model = unet_model.to(device)
    
    if debug:
        print(f"Loaded UNet-enhanced model on {device}")
        print(f"Model has {len(unet_model.transformer.blocks)} layers with "
              f"{unet_model.transformer.blocks[0].attn.num_heads} heads each")
    
    return unet_model