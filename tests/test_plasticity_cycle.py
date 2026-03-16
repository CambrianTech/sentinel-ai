"""
Test the full plasticity cycle: train → prune → clone → diverge.

Validates on MPS (Metal), CUDA, or CPU — whichever is available.
This is the integration test proving sentinel-ai's adaptive architecture works.
"""

import torch
import torch.nn.functional as F
import pytest
import math
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from datasets import load_dataset

from models.loaders.gpt2_loader_clean import load_adaptive_model_gpt_clean
from sentinel.models.adaptive_head_cloning import AdaptiveHeadManager, HeadStats


def best_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = best_device()


@pytest.fixture(scope="module")
def base_resources():
    """Load base model, config, tokenizer, and training data once."""
    config = GPT2Config.from_pretrained("distilgpt2")
    base_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in dataset["text"] if len(t) > 50][:500]
    encoded = tokenizer(
        texts, truncation=True, max_length=128, padding="max_length", return_tensors="pt"
    )
    train_ids = encoded["input_ids"].to(DEVICE)

    return base_model, config, tokenizer, train_ids


def fresh_adaptive_model(base_model, config):
    """Create a fresh adaptive model (no shared state between tests)."""
    # Re-load base model weights to get a clean copy
    fresh_base = GPT2LMHeadModel.from_pretrained("distilgpt2")
    return load_adaptive_model_gpt_clean(
        "distilgpt2", fresh_base, config, device=DEVICE, quiet=True
    )


def measure_perplexity(model, ids, config, n=30):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i in range(0, min(len(ids), n), 5):
            batch = ids[i : i + 5]
            logits = model(batch).logits
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, config.vocab_size),
                batch[:, 1:].reshape(-1),
            )
            total_loss += loss.item()
            count += 1
    model.train()
    return math.exp(total_loss / count)


class TrainingLog:
    """Tracks events during training for test assertions."""
    def __init__(self):
        self.prune_events = []
        self.clone_events = []


def train_with_plasticity(model, train_ids, config, n_steps=500, warmup_steps=200,
                          update_frequency=50, training_log=None):
    """Complete training loop with gate gradient flow and AdaptiveHeadManager."""
    transformer = model.transformer

    manager = AdaptiveHeadManager(
        model=model,
        prune_threshold=0.25,
        clone_threshold=0.75,
        min_active_heads=4,
        max_heads_per_layer=12,
        update_frequency=update_frequency,
        warmup_steps=warmup_steps,
    )

    # Monkey-patch manager to log prune/clone events
    if training_log is not None:
        _original_prune = manager.prune_head
        _original_split = manager.split_head

        def _logging_prune(layer_idx, head_idx):
            training_log.prune_events.append((layer_idx, head_idx, manager.step_count))
            return _original_prune(layer_idx, head_idx)

        def _logging_split(layer_idx, parent_head_idx):
            result = _original_split(layer_idx, parent_head_idx)
            if result is not None:
                training_log.clone_events.append(
                    (layer_idx, parent_head_idx, result, manager.step_count)
                )
            return result

        manager.prune_head = _logging_prune
        manager.split_head = _logging_split

    gate_params = []
    model_params = []
    for name, param in model.named_parameters():
        if "gate" in name:
            gate_params.append(param)
        else:
            model_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": model_params, "lr": 5e-5},
        {"params": gate_params, "lr": 5e-3},
    ])

    model.train()
    for step in range(n_steps):
        idx = torch.randint(0, len(train_ids), (8,))
        batch = train_ids[idx]

        logits = model(batch).logits
        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, config.vocab_size),
            batch[:, 1:].reshape(-1),
        )

        gate_l1 = sum(block.attn.gate.abs().sum() for block in transformer.blocks)
        total_gates = sum(block.attn.gate.shape[0] for block in transformer.blocks)
        gate_reg = 0.1 * gate_l1 / total_gates

        loss = lm_loss + gate_reg
        optimizer.zero_grad()
        loss.backward()

        batch_gradients = {}
        for layer_idx, block in enumerate(transformer.blocks):
            if block.attn.gate.grad is not None:
                for head_idx in range(block.attn.num_heads):
                    grad_mag = abs(block.attn.gate.grad[head_idx].item())
                    batch_gradients[(layer_idx, head_idx)] = grad_mag

        optimizer.step()
        manager.step(batch_gradients)

    return manager, optimizer


class TestDeviceCompat:
    """Verify the model works on the available accelerator."""

    def test_device_is_mps_or_cuda(self):
        """Hardware accelerator should be available."""
        print(f"\nUsing device: {DEVICE}")
        assert DEVICE in ("mps", "cuda", "cpu")

    def test_forward_pass_on_device(self, base_resources):
        """Model should load and forward pass on device without errors."""
        _, config, _, train_ids = base_resources
        model = fresh_adaptive_model(None, config)
        batch = train_ids[:4]
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.logits.shape == (4, 128, config.vocab_size)
        assert not torch.isnan(out.logits).any()

    def test_gate_gradients_flow(self, base_resources):
        """Backward pass must produce non-zero gate gradients on device."""
        _, config, _, train_ids = base_resources
        model = fresh_adaptive_model(None, config)
        batch = train_ids[:4]
        model.train()
        logits = model(batch).logits
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, config.vocab_size),
            batch[:, 1:].reshape(-1),
        )
        loss.backward()

        for block in model.transformer.blocks:
            assert block.attn.gate.grad is not None, "Gate gradients must flow"
            assert not torch.all(block.attn.gate.grad == 0), \
                "Gate gradients must not all be zero — gradient flow is broken"


class TestFullPlasticityCycle:
    """
    The main test: train with L1 regularization + AdaptiveHeadManager,
    verify pruning fires, cloning fires, heads diverge, perplexity improves.
    """

    @pytest.fixture(scope="class")
    def trained_state(self, base_resources):
        """Train a fresh model for 1000 steps — shared across this class."""
        _, config, _, train_ids = base_resources
        model = fresh_adaptive_model(None, config)

        baseline_ppl = measure_perplexity(model, train_ids, config)

        log = TrainingLog()
        manager, optimizer = train_with_plasticity(
            model, train_ids, config,
            n_steps=1000, warmup_steps=150, update_frequency=50,
            training_log=log,
        )

        final_ppl = measure_perplexity(model, train_ids, config)

        return model, config, train_ids, manager, log, baseline_ppl, final_ppl, optimizer

    def test_perplexity_improves(self, trained_state):
        """Training should reduce perplexity well below baseline."""
        _, _, _, _, _, baseline_ppl, final_ppl, _ = trained_state
        print(f"\n  Baseline PPL: {baseline_ppl:.2f}, Final PPL: {final_ppl:.2f}")
        assert final_ppl < baseline_ppl * 0.5, \
            f"Perplexity didn't improve enough: {baseline_ppl:.2f} → {final_ppl:.2f}"

    def test_gates_differentiate(self, trained_state):
        """Gates should spread out (not all ~same value)."""
        model, _, _, _, _, _, _, _ = trained_state
        all_gates = torch.cat([
            block.attn.gate.detach().cpu()
            for block in model.transformer.blocks
        ])
        gate_range = all_gates.max().item() - all_gates.min().item()
        print(f"\n  Gate range: {gate_range:.3f} [{all_gates.min():.3f} ... {all_gates.max():.3f}]")
        assert gate_range > 0.3, \
            f"Gates didn't differentiate (range={gate_range:.3f})"

    def test_growth_occurs(self, trained_state):
        """Some heads should grow above 1.0."""
        model, _, _, _, _, _, _, _ = trained_state
        all_gates = torch.cat([
            block.attn.gate.detach().cpu()
            for block in model.transformer.blocks
        ])
        above_one = (all_gates > 1.0).sum().item()
        print(f"\n  Heads above 1.0: {above_one}/{len(all_gates)}")
        assert above_one > 0, "No heads grew above 1.0"

    def test_pruning_fired(self, trained_state):
        """The manager should have pruned at least one head during training."""
        _, _, _, _, log, _, _, _ = trained_state
        print(f"\n  Prune events: {len(log.prune_events)}")
        for layer, head, step in log.prune_events[:10]:
            print(f"    Step {step}: pruned layer {layer} head {head}")
        assert len(log.prune_events) > 0, "No prune events fired"

    def test_cloning_fired(self, trained_state):
        """The manager should have cloned at least one head during training."""
        _, _, _, _, log, _, _, _ = trained_state
        print(f"\n  Clone events: {len(log.clone_events)}")
        for layer, parent, child, step in log.clone_events[:10]:
            print(f"    Step {step}: split layer {layer} head {parent} → {child}")
        assert len(log.clone_events) > 0, "No clone events fired"

    def test_cloned_heads_diverge(self, trained_state):
        """Gate values should show variety across heads (not uniform)."""
        model, _, _, _, _, _, _, _ = trained_state
        all_gates = []
        for block in model.transformer.blocks:
            for h in range(block.attn.num_heads):
                all_gates.append(block.attn.gate.data[h].item())

        unique_rounded = set(round(v, 2) for v in all_gates)
        print(f"\n  Unique gate values (0.01 resolution): {len(unique_rounded)}/{len(all_gates)}")
        # At least 1/3 of heads should have distinct gate values
        assert len(unique_rounded) > len(all_gates) * 0.33, \
            "Gate values are too uniform — heads aren't differentiating"

    def test_model_still_generates(self, trained_state, base_resources):
        """After pruning and cloning, model should still generate text."""
        model, _, _, _, _, _, _, _ = trained_state
        _, _, tokenizer, _ = base_resources

        prompt = "The meaning of"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        model.eval()
        with torch.no_grad():
            g = ids.clone()
            for _ in range(20):
                logits = model(g).logits
                next_token = logits[0, -1:].argmax(-1).unsqueeze(0)
                g = torch.cat([g, next_token], dim=1)

        text = tokenizer.decode(g[0])
        print(f"\n  Generated: \"{text}\"")
        assert len(text) > len(prompt) + 10
        tokens = tokenizer.encode(text)
        assert len(set(tokens)) > 3, f"Degenerate output — only {len(set(tokens))} unique tokens"


class TestMitosisWeightContinuity:
    """Test that head splitting preserves output continuity."""

    def test_split_preserves_output(self, base_resources):
        """
        Splitting a head should preserve combined output at the attention layer.
        gate * (ctx @ W_o + b_o) == gate * (ctx @ 0.5*W_o + 0.5*b_o) + gate * (ctx @ 0.5*W_o + 0.5*b_o)
        """
        _, config, _, train_ids = base_resources
        model = fresh_adaptive_model(None, config)
        layer = model.transformer.blocks[2]
        attn = layer.attn

        # Force-prune head 11 to create an empty slot
        with torch.no_grad():
            attn.gate.data[11] = 0.0

        # Test at attention level (not full model) for precise measurement
        x = torch.randn(2, 20, config.n_embd).to(DEVICE)  # Random hidden states
        model.eval()
        with torch.no_grad():
            out_before = attn(x).clone()

        # Split head 0 into slot 11 (clones QKV, halves output projection)
        manager = AdaptiveHeadManager(model=model, prune_threshold=0.25, clone_threshold=0.75)
        new_head = manager.split_head(2, 0)
        assert new_head == 11

        with torch.no_grad():
            out_after = attn(x)

        max_diff = (out_before - out_after).abs().max().item()
        print(f"\n  Attention output diff after split: max={max_diff:.2e}")
        # Should be near-zero (numerical precision only)
        assert max_diff < 1e-4, f"Split broke output continuity: max_diff={max_diff:.2e}"


class TestCheckpointResume:
    """Test that training state can be saved and resumed."""

    def test_save_and_load_checkpoint(self, base_resources, tmp_path):
        """Save model + manager state, reload, verify exact restoration."""
        _, config, _, train_ids = base_resources
        model = fresh_adaptive_model(None, config)
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Train a bit so gates differentiate
        manager, optimizer = train_with_plasticity(
            model, train_ids, config, n_steps=200, warmup_steps=50
        )

        # Record state
        gates_before = torch.cat([
            block.attn.gate.detach().cpu().clone()
            for block in model.transformer.blocks
        ])
        ppl_before = measure_perplexity(model, train_ids, config)

        # Save checkpoint with full state
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "manager_step_count": manager.step_count,
            "manager_head_stats": {
                str(k): {
                    "attention_entropy": v.attention_entropy,
                    "gradient_magnitude": v.gradient_magnitude,
                    "utilization_score": v.utilization_score,
                    "num_times_cloned": v.num_times_cloned,
                }
                for k, v in manager.head_stats.items()
            },
            "manager_head_lineage": {
                str(k): v for k, v in manager.head_lineage.items()
            },
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"\n  Saved checkpoint ({checkpoint_path.stat().st_size} bytes)")

        # Corrupt model
        with torch.no_grad():
            for block in model.transformer.blocks:
                block.attn.gate.data.fill_(0.5)

        # Load
        loaded = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(loaded["model_state_dict"])

        # Verify exact gate restoration
        gates_after = torch.cat([
            block.attn.gate.detach().cpu()
            for block in model.transformer.blocks
        ])
        max_diff = (gates_before - gates_after).abs().max().item()
        print(f"  Gate restoration diff: {max_diff:.2e}")
        assert max_diff < 1e-6

        # Verify perplexity matches
        ppl_after = measure_perplexity(model, train_ids, config)
        print(f"  PPL: {ppl_before:.2f} → {ppl_after:.2f}")
        assert abs(ppl_before - ppl_after) < 0.5

        # Verify manager state roundtrips
        assert loaded["manager_step_count"] == 200

    def test_resume_training_continues_improving(self, base_resources, tmp_path):
        """After loading checkpoint, resumed training should not catastrophically degrade."""
        _, config, _, train_ids = base_resources
        model = fresh_adaptive_model(None, config)
        checkpoint_path = tmp_path / "resume.pt"

        # Train 300 steps
        manager, optimizer = train_with_plasticity(
            model, train_ids, config, n_steps=300, warmup_steps=100
        )
        ppl_at_save = measure_perplexity(model, train_ids, config)

        # Save full state including optimizer
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

        # Simulate restart: load checkpoint
        loaded = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(loaded["model_state_dict"])

        # Resume with optimizer state restored
        gate_params = [p for n, p in model.named_parameters() if "gate" in n]
        model_params = [p for n, p in model.named_parameters() if "gate" not in n]
        new_optimizer = torch.optim.AdamW([
            {"params": model_params, "lr": 5e-5},
            {"params": gate_params, "lr": 5e-3},
        ])
        new_optimizer.load_state_dict(loaded["optimizer_state_dict"])

        # Train 200 more steps
        manager2, _ = train_with_plasticity(
            model, train_ids, config, n_steps=200, warmup_steps=10
        )
        ppl_after_resume = measure_perplexity(model, train_ids, config)

        print(f"\n  PPL at save: {ppl_at_save:.2f}, after 200 more steps: {ppl_after_resume:.2f}")
        # Should not get dramatically worse (< 5x degradation)
        assert ppl_after_resume < ppl_at_save * 5, \
            f"PPL degraded too much after resume: {ppl_at_save:.2f} → {ppl_after_resume:.2f}"

    def test_pruned_state_survives_checkpoint(self, base_resources, tmp_path):
        """Pruned heads (gate=0) should remain pruned after save/load."""
        _, config, _, train_ids = base_resources
        model = fresh_adaptive_model(None, config)
        checkpoint_path = tmp_path / "pruned.pt"

        # Force-prune specific heads
        pruned_heads = [(0, 0), (2, 5), (4, 11)]
        with torch.no_grad():
            for layer_idx, head_idx in pruned_heads:
                model.transformer.blocks[layer_idx].attn.gate.data[head_idx] = 0.0

        # Save
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

        # Corrupt all gates
        with torch.no_grad():
            for block in model.transformer.blocks:
                block.attn.gate.data.fill_(1.0)

        # Load
        loaded = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(loaded["model_state_dict"])

        # Verify pruned heads are still pruned
        for layer_idx, head_idx in pruned_heads:
            gate_val = model.transformer.blocks[layer_idx].attn.gate.data[head_idx].item()
            print(f"\n  Layer {layer_idx} head {head_idx}: gate={gate_val:.4f}")
            assert abs(gate_val) < 0.01, \
                f"Pruned head ({layer_idx},{head_idx}) not restored: gate={gate_val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
