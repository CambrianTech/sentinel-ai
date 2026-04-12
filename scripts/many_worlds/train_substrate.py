"""train_substrate.py — Phase A: train the Many-Worlds substrate.

The substrate is a shared continuous Gaussian coordinate space.
Multiple frozen pretrained models project into it and read from it
via learned Project/Read adapter pairs.

Phase A training objective:
  1. Contrastive alignment: same input through different models
     should project to NEARBY substrate coordinates
  2. Round-trip fidelity: project → read should recover the
     original residual stream representation

This is the linker's symbol table — it provides the shared
representation space that makes cross-family expert grafting possible.

Usage:
    python scripts/many_worlds/train_substrate.py \\
        --models Qwen/Qwen2.5-1.5B,meta-llama/Llama-3.2-1B \\
        --corpus calibration/heldout_code300.jsonl \\
        --substrate-dim 256 \\
        --steps 1000 \\
        --output output/substrate_v0/

Or via forge-alloy:
    Included as a stage in a Many-Worlds alloy recipe.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

# Import Many-Worlds primitives
sys.path.insert(0, str(Path(__file__).parent))
from substrate import SubstrateVectorSpace, SubstrateConfig
from project_read import ProjectModule, ReadModule, AdapterPair, AdapterConfig
from losses import contrastive_alignment_loss, round_trip_reconstruction_loss
from cross_attention import SubstrateCrossAttention, SubstrateCrossAttentionHook


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load a frozen pretrained model for substrate training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, tokenizer


def get_model_outputs(model, tokenizer, text: str, layer_idx: int, device: str = "cuda"):
    """Extract hidden states at a layer AND logits from a frozen model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    return hidden.float(), logits.float(), inputs


def get_hidden_states(model, tokenizer, text: str, layer_idx: int, device: str = "cuda"):
    """Extract hidden states at a specific layer from a frozen model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
    # Cast to float32 — models output bfloat16 but adapter layers are float32
    return hidden.mean(dim=1, keepdim=True).float()  # (batch, 1, hidden_dim)


def train_substrate(
    model_names: list[str],
    corpus_path: str,
    substrate_dim: int = 256,
    num_gaussians: int = 128,
    steps: int = 1000,
    learning_rate: float = 1e-4,
    output_dir: str = "output/substrate_v0",
    device: str = "cuda",
):
    """Train the Many-Worlds substrate against multiple frozen models."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MANY-WORLDS SUBSTRATE TRAINING")
    print(f"{'='*60}")
    print(f"Models: {model_names}")
    print(f"Substrate: dim={substrate_dim}, gaussians={num_gaussians}")
    print(f"Steps: {steps}, LR: {learning_rate}")
    print(f"Output: {output_dir}")

    # Load corpus
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", item.get("content", ""))
            if text.strip():
                corpus.append(text)
    print(f"Corpus: {len(corpus)} examples")

    # Load models
    models = {}
    tokenizers = {}
    hidden_dims = {}
    for name in model_names:
        print(f"Loading {name}...")
        model, tokenizer = load_model_and_tokenizer(name, device)
        models[name] = model
        tokenizers[name] = tokenizer
        hidden_dims[name] = model.config.hidden_size
        print(f"  hidden_dim={model.config.hidden_size}, layers={model.config.num_hidden_layers}")

    # Create substrate
    substrate_config = SubstrateConfig(
        dimensionality=substrate_dim,
        num_bases=num_gaussians,
    )
    substrate = SubstrateVectorSpace(substrate_config, device=device)

    # Create per-model adapter pairs
    adapters = {}
    insert_layers = {}
    for name in model_names:
        hidden_dim = hidden_dims[name]
        num_layers = models[name].config.num_hidden_layers
        insert_layer = int(num_layers * 2 / 3)  # 2/3 depth
        insert_layers[name] = insert_layer

        adapter_config = AdapterConfig(
            residual_hidden_size=hidden_dim,
            substrate_dim=substrate_dim,
            lora_rank=substrate_dim,  # rank = substrate dim — adapter shouldn't be the bottleneck
            layer_idx=insert_layer,
        )
        adapters[name] = AdapterPair(adapter_config, base_model_name=name, device=device)
        print(f"  {name}: adapter at layer {insert_layer}")

    # Cross-attention blocks — one per target model in the population.
    # Each learns to attend from that model's residual stream to the substrate field.
    cross_attns = {}
    for name in model_names:
        cross_attns[name] = SubstrateCrossAttention(
            target_hidden_dim=hidden_dims[name],
            substrate_dim=substrate_dim,
            num_heads=4,
        ).to(device)
        ca_params = sum(p.numel() for p in cross_attns[name].parameters())
        print(f"  {name}: cross-attention ({ca_params:,} params, gate init=0)")

    # Optimizer: substrate + adapters + cross-attention
    all_params = list(substrate.parameters())
    for adapter in adapters.values():
        all_params.extend(adapter.parameters())
    for ca in cross_attns.values():
        all_params.extend(ca.parameters())

    optimizer = AdamW(all_params, lr=learning_rate, weight_decay=0.01)

    # Cosine LR schedule with linear warmup — critical for zero-init adapters.
    # The adapters start as no-ops (zero output scale). Warmup lets gradients
    # find direction before cosine decay pulls the LR down.
    warmup_steps = min(steps // 10, 500)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=steps - warmup_steps, eta_min=learning_rate * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    print(f"  LR schedule: warmup {warmup_steps} steps → cosine decay to {learning_rate * 0.01:.1e}")

    # Training loop
    print(f"\nTraining...")
    start_time = time.time()
    losses_history = []
    best_loss = float("inf")
    best_step = 0

    # Knowledge distillation through the substrate — the DIRECT way.
    # For each (source, target) pair on the same input:
    #   1. Source model: forward → hidden states + logits (teacher signal)
    #   2. Source hidden → Project → substrate field (mu)
    #   3. Install cross-attention hook on target model
    #   4. Target model: forward WITH cross-attention active → logits
    #   5. Loss: KL divergence between source and augmented-target logits
    # The gradient flows through: target logits → hook → cross-attn → adapter → substrate
    # This directly optimizes "does the substrate make the target predict better?"
    import torch.nn.functional as F

    temperature = 4.0  # soft targets for distillation (Hinton et al.)

    for step in range(steps):
        text = corpus[step % len(corpus)]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Get source hidden states + logits (frozen, no grad)
        names_list = list(model_names)
        model_outputs = {}
        for name in model_names:
            hidden, logits, inputs = get_model_outputs(
                models[name], tokenizers[name], text,
                insert_layers[name], device
            )
            model_outputs[name] = {"hidden": hidden, "logits": logits, "inputs": inputs}

        # Project all into substrate
        mu_cache = {}
        for name in model_names:
            mu, log_var = adapters[name].project(model_outputs[name]["hidden"])
            mu_cache[name] = mu

        # Cross-attention distillation at the INJECTION POINT (local loss).
        # No frozen layers in the gradient path. The cross-attention delta
        # is applied directly to the target's hidden state at layer L,
        # then projected back into substrate space. The loss rewards
        # the augmented projection for being closer to the source's projection.
        for src_name in names_list:
            for tgt_name in names_list:
                if src_name == tgt_name:
                    continue

                src_mu = mu_cache[src_name]  # (1, seq_src, substrate_dim)
                tgt_hidden = model_outputs[tgt_name]["hidden"]  # (1, seq_tgt, tgt_dim)

                # Cross-attend: target queries source's substrate field
                ca_delta = cross_attns[tgt_name](tgt_hidden, src_mu)
                # ca_delta: (1, seq_tgt, tgt_hidden_dim)

                # What SHOULD the CA delta look like? It should approximate
                # what the Read module produces from the source's substrate
                # projection — that's the "ideal" transfer signal.
                # The Read module maps substrate→target_residual, so it
                # already knows the right shape and magnitude.
                ideal_delta = adapters[tgt_name].read(src_mu).detach()
                # ideal_delta: (1, seq_src, tgt_hidden_dim)

                # Match sequence lengths (pool both)
                ca_pooled = ca_delta.mean(dim=1)      # (1, tgt_hidden_dim)
                ideal_pooled = ideal_delta.mean(dim=1) # (1, tgt_hidden_dim)

                # Loss: CA output should look like the Read module's output
                # This directly teaches the CA to extract the substrate info
                # No zero-init adapters in the path. No frozen model layers.
                transfer_sim = F.cosine_similarity(ca_pooled, ideal_pooled, dim=-1).mean()
                transfer_loss = 1.0 - transfer_sim

                total_loss = total_loss + transfer_loss

        # Contrastive alignment
        projections = {name: mu_cache[name].mean(dim=1) for name in model_names}
        if len(model_names) >= 2:
            align_loss = contrastive_alignment_loss(projections)
            total_loss = total_loss + align_loss

        # Backward + step
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = total_loss.item()
        losses_history.append(loss_val)

        # Track best
        avg_recent = sum(losses_history[-50:]) / min(50, len(losses_history))
        if avg_recent < best_loss:
            best_loss = avg_recent
            best_step = step

        if step % 100 == 0 or step == steps - 1:
            elapsed = time.time() - start_time
            lr_now = scheduler.get_last_lr()[0]
            print(f"  step {step:5d}/{steps} | loss={loss_val:.4f} | avg50={avg_recent:.4f} | best={best_loss:.4f}@{best_step} | lr={lr_now:.2e} | {elapsed:.0f}s")

    # Save substrate + adapters
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.0f}s")

    substrate.save(str(out / "substrate.pt"))
    for name in model_names:
        safe_name = name.replace("/", "_")
        adapters[name].save(str(out / f"adapter_{safe_name}.pt"))
        torch.save(cross_attns[name].state_dict(), out / f"cross_attn_{safe_name}.pt")

    # Save training metadata
    metadata = {
        "models": model_names,
        "substrate_dim": substrate_dim,
        "num_gaussians": num_gaussians,
        "steps": steps,
        "learning_rate": learning_rate,
        "corpus_path": corpus_path,
        "corpus_size": len(corpus),
        "final_loss": losses_history[-1] if losses_history else None,
        "training_time_seconds": elapsed,
        "insert_layers": {n: insert_layers[n] for n in model_names},
        "hidden_dims": {n: hidden_dims[n] for n in model_names},
        "losses_history": losses_history[-100:],  # last 100 for the chart
    }
    (out / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved to {output_dir}")
    print(f"  substrate.pt")
    for name in model_names:
        print(f"  adapter_{name.replace('/', '_')}.pt")
    print(f"  training_metadata.json")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Many-Worlds substrate")
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--corpus", required=True, help="Calibration corpus JSONL")
    parser.add_argument("--substrate-dim", type=int, default=256)
    parser.add_argument("--num-gaussians", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output", default="output/substrate_v0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_substrate(
        model_names=args.models.split(","),
        corpus_path=args.corpus,
        substrate_dim=args.substrate_dim,
        num_gaussians=args.num_gaussians,
        steps=args.steps,
        learning_rate=args.lr,
        output_dir=args.output,
        device=args.device,
    )
