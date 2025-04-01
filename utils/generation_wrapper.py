# utils/generation_wrapper.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, 
                top_p=0.9, top_k=50, repetition_penalty=1.2, device="cuda"):
    """
    Generate text using a model.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for the model
        prompt: Text prompt to start generation
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter (higher = more diverse)
        top_k: Top-k sampling parameter (higher = more diverse)
        repetition_penalty: Penalty for repeating tokens (higher = less repetition)
        device: Device to run generation on
        
    Returns:
        Generated text string
    """
    # Create a wrapper for generation
    wrapper = GenerationWrapper(model=model, tokenizer=tokenizer, device=device)
    
    # Generate text
    outputs = wrapper.generate_text(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=1
    )
    
    # Return the first (and only) generated text
    return outputs[0]

class GenerationWrapper:
    def __init__(self, model_name=None, model=None, tokenizer=None, device="cpu"):
        """
        Initialize generation wrapper with either a model name or explicit model and tokenizer.
        
        Args:
            model_name: Name of the pretrained model (e.g., "gpt2")
            model: Explicit model instance (either this or model_name must be provided)
            tokenizer: Tokenizer instance (required if model is provided)
            device: Device to run generation on
        """
        self.device = device
        
        if model_name is not None:
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        elif model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.model_name = "custom_model"
        else:
            raise ValueError("Either model_name or both model and tokenizer must be provided")
    
    def generate_text(self, prompt, max_length=50, temperature=0.8, top_k=50, top_p=0.95,
                    repetition_penalty=1.0, do_sample=True, num_return_sequences=1,
                    visualize_attention=False, track_gate_values=False, use_beam_search=False,
                    num_beams=5, beam_early_stopping=True):
        """
        Generate text using the model based on a given prompt.
        
        Args:
            prompt: Text prompt to start generation
            max_length: Maximum length of generated text (including prompt)
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability threshold for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (vs greedy decoding)
            num_return_sequences: Number of sequences to generate
            visualize_attention: Whether to visualize attention patterns
            track_gate_values: Whether to track gate values during generation
            
        Returns:
            List of generated text sequences
        """
        # Check if this is a standard HuggingFace model or our adaptive model
        if hasattr(self.model, "generate") and not hasattr(self.model, "blocks"):
            # Standard HuggingFace model
            return self._generate_huggingface(
                prompt, max_length, temperature, top_k, top_p,
                repetition_penalty, do_sample, num_return_sequences
            )
        else:
            # Our adaptive model
            return self._generate_adaptive(
                prompt, max_length, temperature, top_k, top_p,
                repetition_penalty, do_sample, num_return_sequences,
                visualize_attention, track_gate_values, use_beam_search,
                num_beams, beam_early_stopping
            )
    
    def _generate_huggingface(self, prompt, max_length, temperature, top_k, top_p,
                           repetition_penalty, do_sample, num_return_sequences):
        """Generate text using HuggingFace's generation method."""
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences
            )
        
        # Convert outputs to text
        generated_sequences = []
        for output_seq in output:
            generated_sequences.append(self.tokenizer.decode(output_seq, skip_special_tokens=True))
        
        return generated_sequences
    
    def _generate_adaptive(self, prompt, max_length, temperature, top_k, top_p,
                        repetition_penalty, do_sample, num_return_sequences,
                        visualize_attention, track_gate_values, use_beam_search=False, 
                        num_beams=5, beam_early_stopping=True):
        """
        Generate text using our adaptive transformer model with improved quality.
        
        Args:
            prompt: Text prompt to start generation
            max_length: Maximum length of generated text (including prompt)
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability threshold for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (vs greedy decoding)
            num_return_sequences: Number of sequences to generate
            visualize_attention: Whether to visualize attention patterns
            track_gate_values: Whether to track gate values during generation
            use_beam_search: Whether to use beam search for higher quality generation
            num_beams: Number of beams for beam search
            beam_early_stopping: Whether to stop beam search when all beams finished
        """
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Store attention visualizations if requested
        attention_maps = [] if visualize_attention else None
        gate_values_history = [] if track_gate_values else None
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Initialize with input_ids
            current_input = input_ids
            
            # Check if we should use beam search for higher quality generation
            if use_beam_search:
                return self._beam_search_generation(
                    input_ids, max_length, temperature, top_k, top_p,
                    repetition_penalty, num_beams, num_return_sequences,
                    attention_maps, gate_values_history
                )
                
            # Otherwise, proceed with normal autoregressive generation
            # Store the generated tokens for each sequence
            generated_tokens = [[] for _ in range(num_return_sequences)]
            active_sequences = list(range(num_return_sequences))
            
            # Store the initial input length for tracking new tokens
            input_length = input_ids.shape[1]
            
            # Duplicate input for multiple sequences if needed
            if num_return_sequences > 1:
                current_input = current_input.repeat(num_return_sequences, 1)
            
            # Generate tokens sequentially
            for _ in tqdm(range(max_length - input_length), desc="Generating"):
                # Forward pass through the model
                outputs = self.model(current_input)
                
                # Get the next token logits (last position in the sequence)
                # Handle different output formats
                if isinstance(outputs, torch.Tensor):
                    # Direct tensor output
                    next_token_logits = outputs[:, -1, :] / temperature
                else:
                    # CausalLMOutput format
                    next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # Apply enhanced repetition penalty with decay
                if repetition_penalty != 1.0:
                    for i in range(next_token_logits.shape[0]):
                        for idx, token_id in enumerate(current_input[i]):
                            # Calculate position-based decay for penalty
                            # More recent tokens receive stronger penalties
                            position = idx / len(current_input[i])  # 0.0 to 1.0
                            # Scale penalty by position (stronger for recent tokens)
                            position_factor = 0.5 + 0.5 * position
                            # Apply scaled penalty
                            effective_penalty = 1.0 + (repetition_penalty - 1.0) * position_factor
                            next_token_logits[i, token_id] /= effective_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("Inf")
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter back the indices
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float("Inf")
                
                # Enhanced sampling with improved distribution handling
                if do_sample:
                    # Apply a softer softmax with a small amount of extra temperature
                    sampling_temperature = 1.05  # Slightly higher for sampling diversity
                    probs = F.softmax(next_token_logits / sampling_temperature, dim=-1)
                    
                    # Add a small amount of smoothing to avoid degenerate distributions
                    smoothing = 1e-5
                    if smoothing > 0:
                        probs = (1 - smoothing) * probs + smoothing / probs.size(-1)
                        probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normalize
                    
                    # Sample from the distribution
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Capture attention patterns if requested
                if visualize_attention:
                    # Collect attention weights from all layers
                    layer_attentions = []
                    for block in self.model.blocks:
                        if hasattr(block["attn"], "attention_weights") and block["attn"].attention_weights:
                            # Average across heads that have recorded weights
                            head_weights = [w for w in block["attn"].attention_weights.values()]
                            if head_weights:
                                # Get last sequence position attention (what the model focused on)
                                attn_weights = torch.stack(head_weights)  # [num_heads, batch, seq, seq]
                                last_pos_attn = attn_weights[:, 0, -1, :]  # [num_heads, seq]
                                layer_attentions.append(last_pos_attn.cpu())
                    
                    if layer_attentions:
                        attention_maps.append(layer_attentions)
                
                # Track gate values if requested
                if track_gate_values:
                    gates = []
                    for block in self.model.blocks:
                        gates.append(block["attn"].gate.detach().cpu())
                    gate_values_history.append(gates)
                
                # Append the next tokens
                for i, seq_idx in enumerate(active_sequences):
                    generated_tokens[seq_idx].append(next_tokens[i].item())
                
                # Update the input for the next iteration
                current_input = torch.cat([current_input, next_tokens], dim=1)
            
            # Convert the generated tokens to text
            generated_sequences = []
            for i in range(num_return_sequences):
                # Combine the input and generated tokens
                combined_tokens = input_ids[0].tolist() + generated_tokens[i]
                # Decode the tokens to text
                generated_text = self.tokenizer.decode(combined_tokens, skip_special_tokens=True)
                generated_sequences.append(generated_text)
        
        # Create visualizations if requested
        if visualize_attention and attention_maps:
            self._visualize_generation_attention(attention_maps, input_ids[0], generated_tokens[0])
        
        # Create gate value visualization if requested
        if track_gate_values and gate_values_history:
            self._visualize_gate_dynamics(gate_values_history)
        
        return generated_sequences
        
    def _beam_search_generation(self, input_ids, max_length, temperature, top_k, top_p,
                              repetition_penalty, num_beams, num_return_sequences,
                              attention_maps, gate_values_history):
        """
        Implement beam search generation for higher quality outputs.
        
        This is a simplified beam search that maintains multiple sequences and selects
        the best ones based on cumulative scores.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[1]
        vocab_size = self.model.config.vocab_size
        
        # Storage for sequences and scores
        sequences = input_ids.repeat(num_beams, 1)  # [num_beams, seq_len]
        sequence_scores = torch.zeros(num_beams, device=device)  # [num_beams]
        
        # Start with all beams active
        active_beams = list(range(num_beams))
        finished_sequences = []
        finished_scores = []
        
        # Generate until max_length or all beams are finished
        for step in tqdm(range(max_length - input_length), desc="Beam Searching"):
            # Forward pass to get logits for current sequences
            outputs = self.model(sequences[active_beams])
            
            # Get the logits for the next token (last position)
            if isinstance(outputs, torch.Tensor):
                next_token_logits = outputs[:, -1, :] / temperature
            else:
                next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i, seq in enumerate(sequences[active_beams]):
                    for token_id in seq:
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            
            # Apply top-p filtering
            if top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Calculate probabilities
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            # Get top-k tokens and scores for each active beam
            beam_scores = []
            beam_tokens = []
            beam_indices = []
            
            for beam_idx, beam_probs in enumerate(next_token_probs):
                # Get top-2*num_beams tokens for this beam
                top_scores, top_tokens = beam_probs.topk(2 * num_beams, dim=-1)
                
                for token_idx, (score, token) in enumerate(zip(top_scores, top_tokens)):
                    # Calculate combined score (previous + current)
                    combined_score = sequence_scores[active_beams[beam_idx]] + torch.log(score)
                    beam_scores.append(combined_score)
                    beam_tokens.append(token)
                    beam_indices.append(active_beams[beam_idx])
            
            # Get the best 2*num_active_beams candidates
            beam_scores = torch.stack(beam_scores)
            
            # If we don't have enough candidates, pad with very negative scores
            if len(beam_scores) < 2 * len(active_beams):
                padding = torch.ones(2 * len(active_beams) - len(beam_scores), device=device) * -1e9
                beam_scores = torch.cat([beam_scores, padding])
                beam_tokens.extend([0] * (2 * len(active_beams) - len(beam_tokens)))
                beam_indices.extend([0] * (2 * len(active_beams) - len(beam_indices)))
            
            # Get top scores and their indices
            top_scores, top_indices = beam_scores.topk(len(active_beams), dim=0)
            
            # Create new sequences
            new_sequences = []
            new_scores = []
            new_active_beams = []
            
            for i, (score_idx, score) in enumerate(zip(top_indices, top_scores)):
                score_idx = score_idx.item()
                beam_idx = beam_indices[score_idx]
                token = beam_tokens[score_idx]
                
                # Check if this is an EOS token
                if token.item() == self.tokenizer.eos_token_id:
                    # Add to finished sequences
                    finished_sequence = torch.cat([sequences[beam_idx], token.unsqueeze(0)])
                    finished_sequences.append(finished_sequence)
                    finished_scores.append(score)
                else:
                    # Continue this beam
                    new_sequence = torch.cat([sequences[beam_idx], token.unsqueeze(0)])
                    new_sequences.append(new_sequence)
                    new_scores.append(score)
                    new_active_beams.append(i)
            
            # If we have no active beams left, break
            if not new_active_beams:
                break
            
            # Update sequences and scores
            if new_sequences:
                sequences = torch.stack(new_sequences)
                sequence_scores = torch.stack(new_scores)
                active_beams = new_active_beams
            
            # If we have enough finished sequences, break
            if len(finished_sequences) >= num_return_sequences:
                break
        
        # If we don't have enough finished sequences, add the active beams
        while len(finished_sequences) < num_return_sequences and active_beams:
            # Add best active beam to finished sequences
            best_idx = torch.argmax(sequence_scores).item()
            finished_sequences.append(sequences[best_idx])
            finished_scores.append(sequence_scores[best_idx])
            
            # Remove this beam
            sequences = torch.cat([sequences[:best_idx], sequences[best_idx+1:]])
            sequence_scores = torch.cat([sequence_scores[:best_idx], sequence_scores[best_idx+1:]])
            active_beams.pop(0)
        
        # Sort finished sequences by score
        if finished_sequences:
            finished_scores = torch.stack(finished_scores)
            _, sorted_indices = finished_scores.sort(descending=True)
            finished_sequences = [finished_sequences[i] for i in sorted_indices]
        
        # Decode the sequences
        generated_sequences = []
        for seq in finished_sequences[:num_return_sequences]:
            generated_text = self.tokenizer.decode(seq, skip_special_tokens=True)
            generated_sequences.append(generated_text)
        
        # Ensure we have at least num_return_sequences
        while len(generated_sequences) < num_return_sequences:
            generated_sequences.append(generated_sequences[0] if generated_sequences else "")
        
        return generated_sequences
    
    def _top_k_filtering(self, logits, top_k):
        """Apply top-k filtering to logits."""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = -float("Inf")
        return filtered_logits
    
    def _top_p_filtering(self, logits, top_p):
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back the indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = -float("Inf")
        return filtered_logits
    
    def _visualize_generation_attention(self, attention_maps, input_ids, generated_tokens):
        """
        Visualize attention patterns during text generation.
        
        Args:
            attention_maps: List of attention weight tensors
            input_ids: Input token IDs
            generated_tokens: Generated token IDs
        """
        # Combine input and generated tokens
        all_tokens = input_ids.tolist() + generated_tokens
        
        # Decode tokens to text
        token_strings = [self.tokenizer.decode([token]) for token in all_tokens]
        
        # Create a figure for each generation step
        for step, layer_attentions in enumerate(attention_maps):
            step_pos = len(input_ids) + step - 1  # -1 because we look at the attention for the next token
            focus_token = token_strings[step_pos]
            
            # Create a multi-layer visualization
            num_layers = len(layer_attentions)
            plt.figure(figsize=(15, num_layers * 2))
            
            for i, attn in enumerate(layer_attentions):
                # Get attention weights for this layer (average across heads)
                # Limit to show attention up to the current position
                attention_slice = attn[:, :step_pos+1]
                
                # Create subplot
                plt.subplot(num_layers, 1, i+1)
                
                # Create heatmap
                sns.heatmap(
                    attention_slice.numpy(),
                    xticklabels=token_strings[:step_pos+1],
                    yticklabels=[f"Head {j}" for j in range(attn.shape[0])],
                    cmap="viridis"
                )
                
                plt.title(f"Layer {i} Attention for token: '{focus_token}'")
                plt.ylabel("Attention Heads")
                plt.xlabel("Input Tokens")
            
            plt.tight_layout()
            plt.savefig(f"attention_step_{step}.png")
            plt.close()
    
    def _visualize_gate_dynamics(self, gate_values_history):
        """
        Visualize how gate values change during text generation.
        
        Args:
            gate_values_history: List of gate values for each generation step
        """
        num_layers = len(gate_values_history[0])
        
        plt.figure(figsize=(15, num_layers * 2))
        
        for layer_idx in range(num_layers):
            plt.subplot(num_layers, 1, layer_idx + 1)
            
            # Extract gate values for this layer across all steps
            layer_gates = torch.stack([step[layer_idx] for step in gate_values_history])
            
            # Create a plot with one line per head
            num_heads = layer_gates.shape[1]
            for head_idx in range(num_heads):
                head_values = layer_gates[:, head_idx].numpy()
                plt.plot(head_values, label=f"Head {head_idx}")
                
            plt.title(f"Layer {layer_idx} Gate Values During Generation")
            plt.xlabel("Generation Step")
            plt.ylabel("Gate Value")
            plt.ylim(0, 1.1)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig("gate_dynamics.png")
        plt.close()
    
    def run_inference(self, prompts, output_file=None, **generation_kwargs):
        """
        Run inference on multiple prompts and optionally save results.
        
        Args:
            prompts: List of prompt texts
            output_file: Optional file to save results
            **generation_kwargs: Keyword arguments for the generate_text function
            
        Returns:
            Dictionary mapping prompts to generated texts
        """
        results = {}
        
        for prompt in tqdm(prompts, desc="Processing prompts"):
            generated_texts = self.generate_text(
                prompt, **generation_kwargs
            )
            results[prompt] = generated_texts
        
        if output_file:
            with open(output_file, "w") as f:
                for prompt, generations in results.items():
                    f.write(f"Prompt: {prompt}\n\n")
                    for i, text in enumerate(generations):
                        f.write(f"Generation {i+1}:\n{text}\n\n")
                    f.write("-" * 50 + "\n\n")
        
        return results

