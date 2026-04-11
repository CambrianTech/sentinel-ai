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
from substrate import SubstrateVectorSpace
from project_read import ProjectModule, ReadModule, AdapterPair
from losses import contrastive_alignment_loss, round_trip_reconstruction_loss


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


def get_hidden_states(model, tokenizer, text: str, layer_idx: int, device: str = "cuda"):
    """Extract hidden states at a specific layer from a frozen model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is a tuple of (n_layers + 1) tensors
    hidden = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
    # Take the mean across sequence positions for a single vector
    return hidden.mean(dim=1)  # (batch, hidden_dim)


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
    substrate = SubstrateVectorSpace(
        dim=substrate_dim,
        num_gaussians=num_gaussians,
    ).to(device)

    # Create per-model adapter pairs
    adapters = {}
    insert_layers = {}
    for name in model_names:
        hidden_dim = hidden_dims[name]
        num_layers = models[name].config.num_hidden_layers
        insert_layer = int(num_layers * 2 / 3)  # 2/3 depth
        insert_layers[name] = insert_layer

        project = ProjectModule(hidden_dim, substrate_dim).to(device)
        read = ReadModule(substrate_dim, hidden_dim).to(device)
        adapters[name] = AdapterPair(project=project, read=read)
        print(f"  {name}: adapter at layer {insert_layer}")

    # Optimizer: substrate params + all adapter params
    all_params = list(substrate.parameters())
    for adapter in adapters.values():
        all_params.extend(adapter.project.parameters())
        all_params.extend(adapter.read.parameters())

    optimizer = AdamW(all_params, lr=learning_rate)

    # Training loop
    print(f"\nTraining...")
    start_time = time.time()
    losses_history = []

    for step in range(steps):
        # Sample a random text from corpus
        text = corpus[step % len(corpus)]

        # Get hidden states from each model at the insert layer
        hidden_states = {}
        for name in model_names:
            h = get_hidden_states(
                models[name], tokenizers[name], text,
                insert_layers[name], device
            )
            hidden_states[name] = h  # (1, hidden_dim)

        # Phase A losses
        total_loss = torch.tensor(0.0, device=device)

        # 1. Contrastive alignment: different models' projections should be similar
        projections = {}
        for name in model_names:
            mu, log_var = adapters[name].project(hidden_states[name])
            projections[name] = mu  # (1, substrate_dim)

        if len(model_names) >= 2:
            names_list = list(model_names)
            for i in range(len(names_list)):
                for j in range(i + 1, len(names_list)):
                    align_loss = contrastive_alignment_loss(
                        projections[names_list[i]],
                        projections[names_list[j]],
                    )
                    total_loss = total_loss + align_loss

        # 2. Round-trip reconstruction: project → read should recover original
        for name in model_names:
            mu, log_var = adapters[name].project(hidden_states[name])
            reconstructed = adapters[name].read(mu)
            rt_loss = round_trip_reconstruction_loss(
                hidden_states[name], reconstructed
            )
            total_loss = total_loss + rt_loss

        # Backward + step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_val = total_loss.item()
        losses_history.append(loss_val)

        if step % 50 == 0 or step == steps - 1:
            elapsed = time.time() - start_time
            avg_loss = sum(losses_history[-50:]) / min(50, len(losses_history))
            print(f"  step {step:4d}/{steps} | loss={loss_val:.4f} | avg={avg_loss:.4f} | {elapsed:.0f}s")

    # Save substrate + adapters
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.0f}s")

    substrate.save(str(out / "substrate.pt"))
    for name in model_names:
        safe_name = name.replace("/", "_")
        torch.save({
            "project": adapters[name].project.state_dict(),
            "read": adapters[name].read.state_dict(),
            "insert_layer": insert_layers[name],
            "hidden_dim": hidden_dims[name],
        }, out / f"adapter_{safe_name}.pt")

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
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
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
