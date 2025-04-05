import torch
from torch import nn
import math
import time
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

class GatedMultiHeadSelfAttention(nn.Module):
    """
    Implementation of Gated Multi-Head Self-Attention with Sentinel Gates and Agency.
    
    As described in the paper (Section 2.1):
    - Each attention head has its own gating parameter that controls its contribution
    - Gates are initialized close to 1.0 and can be dynamically adjusted by the controller
    - Heads with gates close to zero are effectively pruned during computation
    
    Extended with Agency Layer functionality:
    - Heads can signal internal states like overloaded, misaligned, or withdrawn
    - The system respects these signals during attention computation and controller updates
    - This implements ethical AI principles of consent and agency
    
    Our implementation uses separate parameter matrices for each head to enable
    fine-grained control and potential specialization during training.
    """
    def __init__(self, embed_dim, num_heads, debug=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Create separate parameter matrices for each attention head
        # Each head has its own query, key, value, and output projections
        # This enables more fine-grained control and potential specialization
        self.W_q = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, self.head_dim) / math.sqrt(embed_dim))
            for _ in range(num_heads)
        ])
        
        self.W_k = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, self.head_dim) / math.sqrt(embed_dim))
            for _ in range(num_heads)
        ])
        
        self.W_v = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, self.head_dim) / math.sqrt(embed_dim))
            for _ in range(num_heads)
        ])
        
        self.W_o = nn.ParameterList([
            nn.Parameter(torch.randn(self.head_dim, embed_dim) / math.sqrt(self.head_dim))
            for _ in range(num_heads)
        ])
        
        # Sentinel gates: learnable parameters to control each head's contribution
        # As described in the paper: "Gates are scalar values g_i ∈ (0,1) 
        # that modulate the contribution of each attention head"
        self.gate = nn.Parameter(torch.ones(num_heads))
        
        # Agency tracking for each head
        self.agency_signals = {}  # Will store signals like {head_idx: {"state": state, "consent": True}}
        self.consent_violations = {}  # Track potential violations for analysis
        
        # Flag to store attention weights during forward pass (for visualization)
        self.store_attention_weights = False
        self.attention_weights = {}
        
        # Debug flag
        self.debug = debug
    
    def set_head_state(self, head_idx, state, consent=None):
        """
        Set the agency state for a specific attention head.
        
        Args:
            head_idx: The index of the attention head
            state: One of "active", "overloaded", "misaligned", "withdrawn"
            consent: Boolean indicating whether the head consents to updates
                    (if None, consent state is unchanged)
        """
        if head_idx >= self.num_heads:
            raise ValueError(f"Invalid head index: {head_idx}, max: {self.num_heads-1}")
            
        # Initialize if this is the first signal for this head
        if head_idx not in self.agency_signals:
            self.agency_signals[head_idx] = {
                "state": "active",
                "consent": True,
                "utilization": 0.0,
                "timestamp": time.time()
            }
        
        # Update the state
        self.agency_signals[head_idx]["state"] = state
        self.agency_signals[head_idx]["timestamp"] = time.time()
        
        # Update consent if provided
        if consent is not None:
            self.agency_signals[head_idx]["consent"] = consent
            
        # For withdrawn state, automatically update gate value
        if state == "withdrawn" or (consent is not None and not consent):
            with torch.no_grad():
                self.gate[head_idx] = 0.0
    
    def _log_consent_violation(self, head_idx, violation_type, step=None):
        """Log potential consent violations for analysis."""
        if head_idx not in self.consent_violations:
            self.consent_violations[head_idx] = []
            
        self.consent_violations[head_idx].append({
            "type": violation_type,
            "timestamp": time.time(),
            "step": step
        })
    
    def get_effective_gate(self, head_idx):
        """
        Get the effective gate value for a head, accounting for agency state.
        
        Args:
            head_idx: The index of the attention head
            
        Returns:
            Effective gate value considering agency state
        """
        # Get base gate value
        gate_value = self.gate[head_idx].item()
        
        # Check for agency signals
        if head_idx in self.agency_signals:
            head_state = self.agency_signals[head_idx]
            
            # Withdrawn state or withdrawn consent means zero gate
            if head_state["state"] == "withdrawn" or not head_state["consent"]:
                return 0.0
                
            # Modify gate based on state
            if head_state["state"] == "overloaded":
                # Reduce contribution when overloaded
                return gate_value * 0.5
                
            elif head_state["state"] == "misaligned":
                # Slight reduction for misaligned heads
                return gate_value * 0.8
        
        # Default case: use gate value directly
        return gate_value
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        """
        Forward pass with gated attention computation.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, embed_dim]
            attention_mask: Attention mask [batch_size, 1, 1, seq_len]
            head_mask: Optional mask for specific heads [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            attention_output: Output tensor [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Initialize output
        attention_output = torch.zeros(
            (batch_size, seq_len, self.embed_dim),
            device=hidden_states.device
        )
        
        # Clear attention weights if we're storing them
        if self.store_attention_weights:
            self.attention_weights = {}
        
        # Track maximum attention value for normalization
        max_attention_value = 0.0
        
        # Process each attention head
        for h in range(self.num_heads):
            # Get effective gate value accounting for agency state
            effective_gate = self.get_effective_gate(h)
            
            # Skip computation for effectively pruned heads (gate ≈ 0)
            if effective_gate < 1e-4:
                continue
                
            # Compute query, key, value projections for this head
            q = torch.matmul(hidden_states, self.W_q[h])  # [batch, seq, head_dim]
            k = torch.matmul(hidden_states, self.W_k[h])  # [batch, seq, head_dim]
            v = torch.matmul(hidden_states, self.W_v[h])  # [batch, seq, head_dim]
            
            # Compute attention scores
            attention_scores = torch.matmul(q, k.transpose(-1, -2))  # [batch, seq, seq]
            
            # Scale attention scores
            attention_scores = attention_scores / math.sqrt(self.head_dim)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            # Apply specific head mask if provided
            if head_mask is not None and head_mask[:, h].sum() > 0:
                attention_scores = attention_scores + head_mask[:, h]
            
            # Apply softmax to get attention probabilities
            attention_probs = torch.softmax(attention_scores, dim=-1)
            
            # Store attention weights if requested
            if self.store_attention_weights:
                self.attention_weights[h] = attention_probs.detach()
            
            # Track maximum attention value (for normalization)
            max_attention_value = max(max_attention_value, attention_probs.abs().max().item())
            
            # Apply attention to values
            context = torch.matmul(attention_probs, v)  # [batch, seq, head_dim]
            
            # Project back to embed_dim
            head_output = torch.matmul(context, self.W_o[h])  # [batch, seq, embed_dim]
            
            # Apply gate and add to output
            attention_output = attention_output + effective_gate * head_output
            
            # Update utilization metric for this head (used by agency system)
            if h in self.agency_signals:
                # Calculate average activation level as a proxy for utilization
                activation_level = attention_probs.abs().mean().item()
                
                # Update exponential moving average of utilization
                prev_util = self.agency_signals[h].get("utilization", 0.0)
                alpha = 0.9  # Smoothing factor
                new_util = alpha * prev_util + (1-alpha) * activation_level
                self.agency_signals[h]["utilization"] = new_util
        
        # Optional: normalize the output based on active gates
        # This helps prevent exploding values when many heads are pruned
        active_gates = sum(self.get_effective_gate(h) > 1e-4 for h in range(self.num_heads))
        if active_gates > 0:
            # Only normalize when we have active heads and significant attention values
            if max_attention_value > 1e-4:
                attention_output = attention_output / max(1.0, active_gates / self.num_heads)
        
        return attention_output


class AdaptiveTransformerBlock(nn.Module):
    """
    Implementation of a transformer block with adaptive attention mechanisms.
    
    This implements the core transformer block as described in the paper with:
    - Gated multi-head attention with sentinel gates
    - Optional U-Net style skip connections between layers
    - Two-layer feed-forward network with gated activation
    - Layer normalization and residual connections
    
    Supports both pre-norm and post-norm transformer variants.
    """
    def __init__(self, hidden_size, num_heads, intermediate_size=None, 
                 prenorm=True, activation="gelu", dropout_prob=0.1, debug=False):
        super().__init__()
        
        # Default intermediate size is 4x hidden size if not specified
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
            
        # Layer normalization and attention
        self.prenorm = prenorm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = GatedMultiHeadSelfAttention(hidden_size, num_heads, debug=debug)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout_prob)
        )
        
        # U-Net style skip connection support
        self.use_skip_connection = False
        self.skip_source = None  # Will be set dynamically
        self.skip_scale = 0.1    # Default scaling factor for skip connections
        
    def forward(self, hidden_states, attention_mask=None, head_mask=None, 
                skip_memory=None, return_dict=False):
        """
        Forward pass for the adaptive transformer block.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask tensor
            head_mask: Optional mask for specific heads
            skip_memory: Optional dictionary of layer outputs for skip connections
            return_dict: Whether to return a dictionary with intermediate outputs
            
        Returns:
            Output tensor or dictionary of outputs
        """
        # Save input for residual connection
        residual = hidden_states
        
        # Layer Normalization (pre-norm or post-norm variant)
        if self.prenorm:
            hidden_states = self.norm1(hidden_states)
            
        # Self-attention
        attn_output = self.attn(hidden_states, attention_mask, head_mask)
        attn_output = self.dropout1(attn_output)
        
        # First residual connection
        hidden_states = residual + attn_output
        
        # Apply skip connection if enabled
        if self.use_skip_connection and self.skip_source is not None and skip_memory is not None:
            if self.skip_source in skip_memory:
                skip_input = skip_memory[self.skip_source]
                if skip_input.shape == hidden_states.shape:
                    # U-Net style connection with scaling
                    hidden_states = hidden_states + self.skip_scale * skip_input
        
        # Save for second residual connection
        residual = hidden_states
        
        # Layer Normalization
        if self.prenorm:
            hidden_states = self.norm2(hidden_states)
        else:
            hidden_states = self.norm1(hidden_states)
            
        # Feed-forward network
        ffn_output = self.ffn(hidden_states)
        
        # Second residual connection
        hidden_states = residual + ffn_output
        
        # Final layer norm for post-norm variant
        if not self.prenorm:
            hidden_states = self.norm2(hidden_states)
        
        if return_dict:
            return {
                "hidden_states": hidden_states,
                "attn_output": attn_output
            }
        return hidden_states


class AdaptiveTransformer(nn.Module):
    """
    Implementation of the Adaptive Transformer model as described in the paper.
    
    This model combines all components:
    - Token embeddings and position embeddings
    - Adaptive transformer blocks with gated attention
    - U-Net skip connections between layers
    - Support for pruning and dynamic architecture
    
    It can be initialized from pre-trained models by transferring embeddings
    and initializing attention heads with the transferred weights.
    """
    def __init__(self, config, debug=False):
        super().__init__()
        
        # Save configuration
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 1024)
        self.position_embedding = nn.Embedding(self.max_position_embeddings, config.hidden_size)
        
        # Get parameters from config
        self.hidden_size = config.hidden_size
        self.num_heads = getattr(config, 'num_attention_heads', config.n_head)
        self.num_layers = getattr(config, 'num_hidden_layers', config.n_layer)
        self.intermediate_size = getattr(config, 'intermediate_size', 4 * config.hidden_size)
        self.dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)
        
        # Create transformer blocks
        self.blocks = nn.ModuleList([
            AdaptiveTransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                dropout_prob=self.dropout_prob,
                debug=debug
            ) for _ in range(self.num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # Debug flag
        self.debug = debug
        
    def get_input_embeddings(self):
        """Get the model's token embedding layer."""
        return self.token_embedding
    
    def set_input_embeddings(self, embeddings):
        """Set the model's token embedding layer."""
        self.token_embedding = embeddings
        
    def forward(self, input_ids, attention_mask=None, head_mask=None, 
                position_ids=None, return_dict=True):
        """
        Forward pass for the Adaptive Transformer.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            head_mask: Optional mask for specific heads
            position_ids: Optional position ids
            return_dict: Whether to return a dictionary with outputs
            
        Returns:
            Last hidden states or dictionary of outputs
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Sum embeddings
        hidden_states = token_embeds + position_embeds
        
        # Format attention mask for transformer blocks
        extended_attention_mask = None
        if attention_mask is not None:
            # Create a 4D mask (batch_size, 1, 1, seq_len)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Convert from 0/1 to -10000.0/0.0
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Dictionary to store layer outputs for skip connections
        layer_outputs = {}
        
        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            # Save output for potential skip connection
            layer_outputs[i] = hidden_states
            
            # Forward through the block
            hidden_states = block(
                hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                skip_memory=layer_outputs
            )
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "hidden_states": layer_outputs
            }
        return hidden_states


class AdaptiveCausalLmWrapper(nn.Module, GenerationMixin):
    """
    Wrapper for the Adaptive Transformer that adds a language modeling head
    and integrates with HuggingFace's generation utilities.
    
    This enables seamless use with HuggingFace's generate() method while
    providing adaptive architecture benefits.
    """
    def __init__(self, base_model, transformer, config):
        super().__init__()
        
        # Store components
        self.base_model = base_model  # Original pretrained model
        self.transformer = transformer  # Adaptive transformer
        self.config = config
        
        # Get the language modeling head from the base model
        if hasattr(base_model, 'lm_head'):
            self.lm_head = base_model.lm_head
        elif hasattr(base_model, 'transformer'):
            if hasattr(base_model.transformer, 'wte'):
                # For GPT-2 style models, often the embedding is tied to output
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                self.lm_head.weight = base_model.transformer.wte.weight
        else:
            # Fallback: create new language model head
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # For generation
        self.main_input_name = "input_ids"
        
    @property
    def device(self):
        """Get the model's device."""
        return next(self.parameters()).device
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass for the language model.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional target token ids for language modeling loss
            
        Returns:
            CausalLMOutput object with logits and optional loss
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        
        hidden_states = transformer_outputs["last_hidden_state"]
        
        # Apply language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.get("hidden_states", None)
        )
    
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        """
        Prepare inputs for generation.
        
        This implements the required method for HuggingFace's GenerationMixin.
        
        Args:
            input_ids: Input token ids
            past: Past key values for efficient generation
            
        Returns:
            Dictionary of model inputs for the next generation step
        """
        # Only last token for inputs_ids if past is defined
        if past is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
        attention_mask = kwargs.get("attention_mask", None)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past
        }
    
    def set_active_adapters(self, adapter_names):
        """
        Set active adapters if the base model supports it.
        
        This is for compatibility with adapter-enabled models.
        
        Args:
            adapter_names: List of adapter names to activate
        """
        if hasattr(self.base_model, "set_active_adapters"):
            self.base_model.set_active_adapters(adapter_names)
    
    def enable_unet_connections(self, enable=True, layer_indices=None):
        """
        Enable or disable U-Net style skip connections.
        
        Args:
            enable: Whether to enable the connections
            layer_indices: Optional list of layer indices to configure
                          If None, configures all decoder-side layers
        """
        if not hasattr(self.transformer, 'blocks'):
            return
            
        num_layers = len(self.transformer.blocks)
        midpoint = num_layers // 2
        
        # Default to configuring all decoder-side layers
        if layer_indices is None:
            layer_indices = list(range(midpoint, num_layers))
            
        for idx in layer_indices:
            if 0 <= idx < num_layers:
                # Enable/disable skip connection
                self.transformer.blocks[idx].use_skip_connection = enable
                
                if enable:
                    # Link to corresponding encoder layer (symmetrically)
                    encoder_idx = num_layers - idx - 1
                    if encoder_idx >= 0:
                        self.transformer.blocks[idx].skip_source = encoder_idx
                
                # Clear for consistency when disabling
                elif not enable:
                    self.transformer.blocks[idx].skip_source = None