"""modeling_avengers.py — HuggingFace-compatible Many-Worlds population model.

Makes a population of frozen models look like one model to the outside world.
Users just do:
    model = AutoModelForCausalLM.from_pretrained("continuum-ai/avengers-v1", trust_remote_code=True)
    output = model.generate(input_ids, max_new_tokens=200)

Internally:
    1. Each source model runs on the input → hidden states → adapter → substrate field
    2. Q-Former queries attend to all sources' substrate fields simultaneously
    3. Confidence gate controls contribution (high when helpful, low when not)
    4. Soft tokens prepended to target model's input as vocab-grounded embeddings
    5. Target model generates with the extra knowledge from the population

Source models are loaded sequentially to minimize VRAM (only one source in memory
at a time during substrate field computation, all freed before target generates).
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    GenerationMixin,
)


class AvengersConfig(PretrainedConfig):
    model_type = "avengers"

    def __init__(
        self,
        sources: list[str] = None,
        target: str = "microsoft/phi-3-mini-4k-instruct",
        substrate_dim: int = 256,
        num_queries: int = 16,
        source_extract_layers: dict = None,
        hidden_dims: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sources = sources or []
        self.target = target
        self.substrate_dim = substrate_dim
        self.num_queries = num_queries
        self.source_extract_layers = source_extract_layers or {}
        self.hidden_dims = hidden_dims or {}


class AvengersModel(PreTrainedModel):
    """A Many-Worlds population that looks like one model.

    Loads source models on-demand, computes substrate fields,
    runs Q-Former, prepends soft tokens, generates from target.
    """
    config_class = AvengersConfig

    def __init__(self, config: AvengersConfig):
        super().__init__(config)
        self.config = config

        # These get loaded from the saved artifacts
        self.qformer = None
        self.src_adapters = {}
        self.target_model = None
        self.target_tok = None
        self.embed_layer = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load the Avengers population from a directory or HF repo."""
        path = Path(pretrained_model_name_or_path)

        # Load metadata
        meta_path = path / "training_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            raise ValueError(f"No training_metadata.json found in {path}")

        # Build config from metadata
        config = AvengersConfig(
            sources=meta.get("sources", meta.get("models", [])[:-1]),
            target=meta.get("target", meta.get("models", [])[-1]),
            substrate_dim=meta["substrate_dim"],
            num_queries=meta["num_queries"],
            source_extract_layers=meta.get("source_extract_layers", {}),
            hidden_dims=meta.get("hidden_dims", {}),
        )

        model = cls(config)
        model._artifacts_path = path
        model._load_components()
        return model

    def _load_components(self):
        """Load Q-Former, adapters, and target model."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from qformer import SubstrateQFormer
        from project_read import AdapterPair

        path = self._artifacts_path
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load target model (stays in memory for generation)
        print(f"Loading target: {self.config.target}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.config.target, torch_dtype=torch.bfloat16, device_map=device)
        self.target_model.eval()
        self.target_tok = AutoTokenizer.from_pretrained(self.config.target)
        self.target_tok.pad_token = self.target_tok.pad_token or self.target_tok.eos_token

        # Find embedding layer
        for name, mod in self.target_model.named_modules():
            if isinstance(mod, nn.Embedding) and mod.weight.shape[0] > 1000:
                self.embed_layer = mod
                break

        tgt_dim = self.target_model.config.hidden_size

        # Load Q-Former
        self.qformer = SubstrateQFormer(
            substrate_dim=self.config.substrate_dim,
            target_embed_dim=tgt_dim,
            num_queries=self.config.num_queries,
        ).to(device)
        self.qformer.set_embedding_table(self.embed_layer.weight)
        self.qformer.set_target_model(self.target_model, self.target_tok, self.embed_layer)
        self.qformer.load_state_dict(
            torch.load(path / "qformer.pt", map_location=device, weights_only=True))
        self.qformer.eval()

        # Load source adapters (models loaded on-demand during inference)
        for sname in self.config.sources:
            safe = sname.replace("/", "_")
            adapter_path = path / f"adapter_{safe}.pt"
            if adapter_path.exists():
                adapter = AdapterPair.load(str(adapter_path), device=device)
                self.src_adapters[sname] = adapter

        print(f"Avengers loaded: {len(self.src_adapters)} sources → {self.config.target}")

    def compute_substrate_fields(self, input_text: str) -> list[torch.Tensor]:
        """Run each source model on input, project into substrate space.

        Sources are loaded and freed SEQUENTIALLY to minimize VRAM.
        Only one source model in memory at a time.
        """
        import gc
        device = next(self.target_model.parameters()).device
        fields = []

        for sname in self.config.sources:
            if sname not in self.src_adapters:
                continue

            # Load source model temporarily
            sm = AutoModelForCausalLM.from_pretrained(
                sname, torch_dtype=torch.bfloat16, device_map=device)
            sm.eval()
            stok = AutoTokenizer.from_pretrained(sname)
            stok.pad_token = stok.pad_token or stok.eos_token

            extract_layer = self.config.source_extract_layers.get(
                sname, int(sm.config.num_hidden_layers * 2 / 3))

            # Forward pass → hidden states → adapter → substrate field
            inputs = stok(input_text, return_tensors="pt",
                         truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = sm(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[extract_layer].float()
            mu, _ = self.src_adapters[sname].project(hidden)
            fields.append(mu.detach())

            # Free source model immediately
            del sm, outputs, hidden
            gc.collect()
            torch.cuda.empty_cache()

        return fields

    def generate(self, input_ids=None, input_text=None, max_new_tokens=200,
                 do_sample=False, **kwargs):
        """Generate with the full Avengers population.

        Accepts either input_ids or raw text. Returns generated token IDs.
        """
        device = next(self.target_model.parameters()).device

        # Get input text for source models
        if input_text is None:
            input_text = self.target_tok.decode(input_ids[0], skip_special_tokens=True)

        # Compute substrate fields from all sources
        fields = self.compute_substrate_fields(input_text)

        # Target input IDs (needed for confidence gating)
        if input_ids is None:
            input_ids = self.target_tok(input_text, return_tensors="pt",
                                       truncation=True, max_length=1024).to(device)["input_ids"]

        # Q-Former: queries attend to all source fields
        # Passes target_input_ids so the confidence gate can measure
        # the target model's uncertainty on this specific input
        soft_tokens = self.qformer(fields, target_input_ids=input_ids)

        with torch.no_grad():
            real_embeds = self.embed_layer(input_ids)

        # Combine soft tokens + real embeddings
        combined = torch.cat([soft_tokens.to(real_embeds.dtype), real_embeds], dim=1)

        # Manual generation loop (HF generate() broken with inputs_embeds)
        generated = []
        past = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if past is None:
                    out = self.target_model(inputs_embeds=combined, use_cache=True)
                    past = out.past_key_values
                else:
                    new_emb = self.embed_layer(
                        torch.tensor([[generated[-1]]], device=device))
                    out = self.target_model(
                        inputs_embeds=new_emb, past_key_values=past, use_cache=True)
                    past = out.past_key_values

                if do_sample:
                    probs = torch.softmax(out.logits[0, -1] / kwargs.get("temperature", 1.0), dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                else:
                    next_id = out.logits[0, -1].argmax().item()

                generated.append(next_id)
                if next_id == self.target_tok.eos_token_id:
                    break

        return torch.tensor([generated], device=device)

    def generate_text(self, prompt: str, max_new_tokens=200, **kwargs) -> str:
        """Convenience method: text in, text out."""
        output_ids = self.generate(input_text=prompt, max_new_tokens=max_new_tokens, **kwargs)
        return self.target_tok.decode(output_ids[0], skip_special_tokens=True)
