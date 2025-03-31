import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from models.adaptive_transformer import AdaptiveTransformerModel

class AdaptiveCausalLmWrapper(AdaptiveTransformerModel, GenerationMixin):
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False

    def __init__(self, config, token_embeddings, position_embeddings):
        super().__init__(config, token_embeddings, position_embeddings)
        self.config = config
        self.generation_config = GenerationConfig.from_model_config(config)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        logits = super().forward(input_ids, attention_mask=attention_mask)
        return CausalLMOutput(logits=logits)

    def can_generate(self):
        return True

    @property
    def _supports_cache_class(self):
        return False

    @property
    def device(self):
        return next(self.parameters()).device


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
