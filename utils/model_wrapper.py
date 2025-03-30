# utils/model_wrapper.py

import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class HuggingFaceWrapper(nn.Module):
    """
    Wraps a custom model to support Hugging Face's `.generate()` API.
    Ensures return type and interface match expected behavior.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, **kwargs):
        logits = self.model(input_ids)
        return CausalLMOutputWithPast(logits=logits)
