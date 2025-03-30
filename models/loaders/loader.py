import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from models.adaptive_transformer import AdaptiveTransformerModel

# This wrapper enables Hugging Face's `.generate()` for custom model
class AdaptiveCausalLmWrapper(AdaptiveTransformerModel, GenerationMixin):
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def forward(self, input_ids, **kwargs):
        logits = super().forward(input_ids)
        return CausalLMOutput(logits=logits)

def load_baseline_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    print(f"✅ Loaded baseline model: {model_name} with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def load_adaptive_model(model_name, baseline_model, device):
    config = AutoConfig.from_pretrained(model_name)
    model = AdaptiveCausalLmWrapper(config, baseline_model.get_input_embeddings(), baseline_model.get_input_embeddings()).to(device)
    baseline_state = baseline_model.state_dict()
    
    loaded_params = []
    adaptive_state = model.state_dict()
    for name, param in adaptive_state.items():
        if name in baseline_state and param.shape == baseline_state[name].shape:
            param.data.copy_(baseline_state[name])
            loaded_params.append(name)
        else:
            print(f"⚠️ Skipped loading param: {name} (shape mismatch or missing in baseline)")

    print(f"✅ Adaptive model initialized from {model_name} weights ({len(loaded_params)}/{len(adaptive_state)} parameters loaded)")
    return model
