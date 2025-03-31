# utils/model_wrapper.py

import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin


class HuggingFaceWrapper(nn.Module, GenerationMixin):
    """
    Wraps a custom model to support Hugging Face's `.generate()` API.
    Ensures return type and interface match expected behavior.
    """
    def __init__(self, model, tokenizer=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = getattr(model, 'config', None)
        self.main_input_name = "input_ids"
        
        # Add generation config if not present and config exists
        if hasattr(self, 'config') and self.config is not None:
            if not hasattr(self, 'generation_config'):
                from transformers import GenerationConfig
                self.generation_config = GenerationConfig.from_model_config(self.config)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        logits = self.model(input_ids, attention_mask=attention_mask)
        return CausalLMOutputWithPast(logits=logits)
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation"""
        return {"input_ids": input_ids}
    
    @property
    def device(self):
        """Get the device where the model is located"""
        return next(self.parameters()).device


def wrap_model_for_generation(model, tokenizer=None):
    """
    Ensure the model has all necessary methods for text generation.
    
    Args:
        model: The model to wrap
        tokenizer: Optional tokenizer for the model
    
    Returns:
        Model with generation capabilities
    """
    # If model already has generate method, return it as is
    if hasattr(model, 'generate') and callable(model.generate):
        return model
    
    # Otherwise, wrap it with our HuggingFaceWrapper
    return HuggingFaceWrapper(model, tokenizer)