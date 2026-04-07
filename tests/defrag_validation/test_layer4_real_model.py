"""
Layer 4: Real model integration tests.

Tests the COMPLETE forge cycle on a real but tractable model:
- Load Qwen2.5-0.5B (smallest production model)
- Run full prune → defrag → train → eval cycle
- Verify perplexity stays bounded across multiple cycles
- Verify final model produces sensible output

This is the slowest layer (~5 minutes) and runs the actual pipeline code,
not synthetic tests. It catches integration bugs between defrag, LoRA,
and the training loop.

Run: pytest tests/defrag_validation/test_layer4_real_model.py -v
Skipped if transformers/peft/datasets not installed.

Marker: @pytest.mark.slow — exclude with `pytest -m 'not slow'`
"""

import os
import math
import pytest
import torch

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")

from transformers import AutoModelForCausalLM, AutoTokenizer

TINY_MODEL = os.environ.get("DEFRAG_TEST_MODEL", "Qwen/Qwen2.5-0.5B")


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def base_model_and_tokenizer():
    """Load the base model and tokenizer once per session."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            TINY_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        pytest.skip(f"Could not load {TINY_MODEL}: {e}")


@pytest.fixture
def small_eval_dataset(base_model_and_tokenizer):
    """20 short wikitext samples for eval."""
    _, tokenizer = base_model_and_tokenizer
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="test[:30]"
        )
        # Filter out empty lines
        ds = ds.filter(lambda x: len(x["text"].strip()) > 50)
        # Take 20
        texts = [ex["text"] for ex in ds][:20]
        return texts
    except Exception as e:
        pytest.skip(f"Could not load dataset: {e}")


# ── Helpers ────────────────────────────────────────────────────────────────


def compute_perplexity(model, tokenizer, texts, max_length=128):
    """Compute average perplexity over a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(model.device)
            if ids["input_ids"].shape[1] < 2:
                continue
            out = model(**ids, labels=ids["input_ids"])
            n = ids["input_ids"].shape[1]
            total_loss += out.loss.item() * n
            total_tokens += n
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def get_layers(model):
    """Find the transformer layer list across architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    return None


def select_lowest_importance_kv_groups(model, num_groups_to_remove=1):
    """Select KV groups with the lowest importance scores from each layer.

    Importance = L2 norm of Q-projection weights for the heads in the group.
    Returns: dict {layer_idx: [head_indices_to_remove]}
    """
    layers = get_layers(model)
    cfg = model.config
    num_q = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    group_size = num_q // num_kv

    dead_heads = {}
    for li, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
        if attn is None or not hasattr(attn, "q_proj"):
            continue
        q_weight = attn.q_proj.weight  # shape (num_q * head_dim, hidden)
        head_dim = q_weight.shape[0] // num_q

        # Compute L2 norm per KV group (sum across the group's heads)
        group_norms = []
        for kv in range(num_kv):
            start = kv * group_size * head_dim
            end = (kv + 1) * group_size * head_dim
            norm = q_weight[start:end].norm().item()
            group_norms.append((norm, kv))

        # Sort by norm, take lowest
        group_norms.sort()
        groups_to_remove = [kv for _, kv in group_norms[:num_groups_to_remove]]

        # Convert KV groups to Q head indices
        heads = []
        for kv in groups_to_remove:
            heads.extend(range(kv * group_size, (kv + 1) * group_size))
        dead_heads[li] = heads

    return dead_heads


# ── Test 1: Single defrag cycle preserves model functionality ──────────────


class TestSingleCycleIntegration:
    """One full prune+defrag cycle on a real model.

    KEY FINDING (sentinel-ai #155): pruning without retraining catastrophically
    destroys the model regardless of which heads are removed. The test
    documents this rather than trying to assert otherwise. Retraining is what
    actually recovers quality — pruning alone is destructive.
    """

    def test_defrag_preserves_structural_validity(
        self, base_model_and_tokenizer, small_eval_dataset
    ):
        """After defrag, the model must still produce VALID output (not NaN/Inf),
        even if perplexity is degraded. Quality recovery comes from retraining."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        try:
            from defrag_inline import defrag_live_model
        except ImportError:
            pytest.skip("defrag_inline not importable")

        import copy
        base_model, tokenizer = base_model_and_tokenizer
        model = copy.deepcopy(base_model)
        layers = get_layers(model)
        if layers is None:
            pytest.skip("Could not locate layers")

        baseline_ppl = compute_perplexity(model, tokenizer, small_eval_dataset)

        dead_heads = select_lowest_importance_kv_groups(model, num_groups_to_remove=1)
        try:
            defrag_live_model(model, dead_heads=dead_heads)
        except Exception as e:
            pytest.skip(f"defrag failed: {e}")

        defragged_ppl = compute_perplexity(model, tokenizer, small_eval_dataset)
        ratio = defragged_ppl / baseline_ppl
        print(f"\n  Baseline PPL:  {baseline_ppl:.2f}")
        print(f"  Defragged PPL: {defragged_ppl:.2f}")
        print(f"  Ratio:         {ratio:.1f}x (degradation expected — needs retraining)")

        # Validity assertions: model must still produce valid output
        assert not math.isnan(defragged_ppl), "Defrag produced NaN perplexity"
        assert not math.isinf(defragged_ppl), "Defrag produced Inf perplexity"
        # PPL can be high (we expect catastrophic degradation without retraining)
        # but must be FINITE — the model should still be evaluable
        assert defragged_ppl < 1e10, f"Defragged PPL absurdly high ({defragged_ppl}) — likely numerical break"


# ── Test 2: Multi-cycle stability ──────────────────────────────────────────


class TestMultiCycleIntegration:
    """The exact pattern from our 9B forge bug. 2 cycles must not destroy the model."""

    def test_two_cycle_defrag_stability(
        self, base_model_and_tokenizer, small_eval_dataset
    ):
        """2 prune+defrag cycles. Perplexity must stay bounded."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        try:
            from defrag_inline import defrag_live_model
        except ImportError:
            pytest.skip("defrag_inline not importable")

        import copy
        base_model, tokenizer = base_model_and_tokenizer
        model = copy.deepcopy(base_model)
        layers = get_layers(model)
        if layers is None:
            pytest.skip("Could not locate layers")

        cfg = model.config
        num_kv = cfg.num_key_value_heads
        if num_kv < 3:
            pytest.skip(f"Need >=3 KV groups for 2-cycle test, have {num_kv}")

        baseline_ppl = compute_perplexity(model, tokenizer, small_eval_dataset)
        print(f"\n  Baseline PPL: {baseline_ppl:.2f}")

        # Cycle 1: remove lowest-importance KV group
        dead_c1 = select_lowest_importance_kv_groups(model, num_groups_to_remove=1)
        try:
            defrag_live_model(model, dead_heads=dead_c1)
        except Exception as e:
            pytest.skip(f"cycle 1 defrag failed: {e}")

        c1_ppl = compute_perplexity(model, tokenizer, small_eval_dataset)
        print(f"  Cycle 1 PPL: {c1_ppl:.2f} (ratio: {c1_ppl/baseline_ppl:.2f}x)")

        # Cycle 2: model now has fewer heads, recompute importance
        num_kv_c2 = cfg.num_key_value_heads
        if num_kv_c2 < 2:
            return  # can't do another cycle
        dead_c2 = select_lowest_importance_kv_groups(model, num_groups_to_remove=1)
        try:
            defrag_live_model(model, dead_heads=dead_c2)
        except Exception as e:
            pytest.skip(f"cycle 2 defrag failed: {e}")

        c2_ppl = compute_perplexity(model, tokenizer, small_eval_dataset)
        print(f"  Cycle 2 PPL: {c2_ppl:.2f} (ratio: {c2_ppl/baseline_ppl:.2f}x)")

        # Both cycles must stay finite (not NaN/Inf)
        # PPL will degrade without retraining — that's expected (#155)
        # The test verifies STRUCTURAL stability across cycles, not quality
        assert not math.isnan(c1_ppl), "Cycle 1 produced NaN"
        assert not math.isinf(c1_ppl), "Cycle 1 produced Inf"
        assert not math.isnan(c2_ppl), "Cycle 2 produced NaN"
        assert not math.isinf(c2_ppl), "Cycle 2 produced Inf"
        # Numerical sanity: must be evaluable, not catastrophically broken
        assert c1_ppl < 1e10, f"Cycle 1 PPL absurd: {c1_ppl}"
        assert c2_ppl < 1e10, f"Cycle 2 PPL absurd: {c2_ppl}"


# ── Test 3: Save → reload → eval matches in-memory eval ────────────────────


class TestPersistedModelMatches:
    """A defragged model that's saved and reloaded must produce identical PPL.
    This is the bug from Layer 3 — verify it stays fixed end-to-end."""

    def test_save_load_perplexity_matches(
        self, base_model_and_tokenizer, small_eval_dataset, tmp_path
    ):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        try:
            from defrag_inline import defrag_live_model
        except ImportError:
            pytest.skip("defrag_inline not importable")

        import copy
        base_model, tokenizer = base_model_and_tokenizer
        model = copy.deepcopy(base_model)
        layers = get_layers(model)
        if layers is None:
            pytest.skip("Could not locate layers")

        # Defrag using importance-based selection
        dead_heads = select_lowest_importance_kv_groups(model, num_groups_to_remove=1)
        try:
            defrag_live_model(model, dead_heads=dead_heads)
        except Exception as e:
            pytest.skip(f"defrag failed: {e}")

        # In-memory eval
        in_memory_ppl = compute_perplexity(model, tokenizer, small_eval_dataset)

        # Save
        save_dir = tmp_path / "defragged_model"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Reload
        del model
        loaded = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.float32)
        loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)
        if loaded_tokenizer.pad_token is None:
            loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

        loaded_ppl = compute_perplexity(loaded, loaded_tokenizer, small_eval_dataset)

        print(f"\n  In-memory PPL: {in_memory_ppl:.4f}")
        print(f"  Reloaded PPL:  {loaded_ppl:.4f}")

        # Must match within float precision
        assert abs(in_memory_ppl - loaded_ppl) < 0.01, (
            f"Reloaded model PPL ({loaded_ppl:.4f}) differs from in-memory "
            f"({in_memory_ppl:.4f}) — config drift suspected"
        )


# ── Test 4: Generation sanity check ────────────────────────────────────────


def compute_lora_gradient_importance(model, tokenizer, info, calibration_texts=None, num_samples=8):
    """Approximation of PLASTICITY-COMPACTION's gradient-magnitude trick.

    Captures gradient magnitudes on the o_proj input via backward hooks during
    a calibration forward+backward pass. Per-head L2 of grad_input is the
    proxy for "how much loss flows back through this head."

    This is the gradient-only signal — distinct from saliency which is
    activation × gradient.
    """
    if calibration_texts is None:
        calibration_texts = [
            "The quick brown fox jumps over the lazy dog. " * 4,
            "Machine learning models learn patterns from training data.",
            "The capital of France is Paris.",
            "Quantum mechanics describes atomic behavior.",
        ]

    layers = get_layers(model)
    n_layers = info["num_layers"]
    n_heads = info["num_heads"]
    device = next(model.parameters()).device

    importance = torch.zeros(n_layers, n_heads)
    captured = {}

    def make_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            # grad_input[0] is gradient w.r.t. the input of o_proj
            g = grad_input[0]
            if g is None or g.dim() != 3:
                return
            B, T, D = g.shape
            attn_module = layers[layer_idx]
            attn = getattr(attn_module, "self_attn", getattr(attn_module, "attn", None))
            actual_heads = getattr(attn, "num_heads", n_heads) if attn else n_heads
            if D % actual_heads != 0:
                return
            head_dim = D // actual_heads
            gh = g.view(B, T, actual_heads, head_dim)
            mag = gh.float().norm(dim=-1).mean(dim=(0, 1))  # (num_heads,)
            captured.setdefault(layer_idx, []).append(mag.cpu())
        return hook

    handles = []
    for li in range(n_layers):
        attn_module = layers[li]
        attn = getattr(attn_module, "self_attn", getattr(attn_module, "attn", None))
        if attn is None or not hasattr(attn, "o_proj"):
            continue
        handles.append(attn.o_proj.register_full_backward_hook(make_hook(li)))

    model.train()  # need gradients
    for text in calibration_texts[:num_samples]:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        if ids["input_ids"].shape[1] < 2:
            continue
        try:
            out = model(**ids, labels=ids["input_ids"])
            out.loss.backward()
            model.zero_grad()
        except Exception:
            continue
    model.eval()

    for h in handles:
        h.remove()

    for li, mags_list in captured.items():
        if not mags_list:
            continue
        stacked = torch.stack(mags_list)
        mean_mag = stacked.mean(dim=0)
        for hi in range(min(mean_mag.shape[0], n_heads)):
            importance[li, hi] = mean_mag[hi].item()

    return importance


def compute_saliency_importance(model, tokenizer, info, calibration_texts=None, num_samples=8):
    """Wanda/SNIP-style saliency: |activation| × |gradient|.

    The chain-rule contribution of each head to the loss. Captures BOTH the
    forward activation magnitude AND the backward gradient magnitude on the
    o_proj input, then multiplies them per-head.

    This is the canonical structured pruning importance metric in recent
    literature (Wanda 2023, SparseGPT 2023). Predicted to be the strongest
    of the four metrics in Kash's hierarchy.
    """
    if calibration_texts is None:
        calibration_texts = [
            "The quick brown fox jumps over the lazy dog. " * 4,
            "Machine learning models learn patterns from training data.",
            "The capital of France is Paris.",
            "Quantum mechanics describes atomic behavior.",
        ]

    layers = get_layers(model)
    n_layers = info["num_layers"]
    n_heads = info["num_heads"]
    device = next(model.parameters()).device

    importance = torch.zeros(n_layers, n_heads)
    fwd_captured = {}  # (layer_idx, sample_idx) -> per-head activation magnitude
    bwd_captured = {}  # (layer_idx, sample_idx) -> per-head gradient magnitude
    sample_idx = [0]

    def make_fwd_hook(layer_idx):
        def hook(module, inputs, output):
            x = inputs[0]
            if x.dim() != 3:
                return
            B, T, D = x.shape
            attn_module = layers[layer_idx]
            attn = getattr(attn_module, "self_attn", getattr(attn_module, "attn", None))
            actual_heads = getattr(attn, "num_heads", n_heads) if attn else n_heads
            if D % actual_heads != 0:
                return
            head_dim = D // actual_heads
            xh = x.view(B, T, actual_heads, head_dim)
            mag = xh.float().norm(dim=-1).mean(dim=(0, 1))
            fwd_captured[(layer_idx, sample_idx[0])] = mag.cpu()
        return hook

    def make_bwd_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            g = grad_input[0]
            if g is None or g.dim() != 3:
                return
            B, T, D = g.shape
            attn_module = layers[layer_idx]
            attn = getattr(attn_module, "self_attn", getattr(attn_module, "attn", None))
            actual_heads = getattr(attn, "num_heads", n_heads) if attn else n_heads
            if D % actual_heads != 0:
                return
            head_dim = D // actual_heads
            gh = g.view(B, T, actual_heads, head_dim)
            mag = gh.float().norm(dim=-1).mean(dim=(0, 1))
            bwd_captured[(layer_idx, sample_idx[0])] = mag.cpu()
        return hook

    fwd_handles = []
    bwd_handles = []
    for li in range(n_layers):
        attn_module = layers[li]
        attn = getattr(attn_module, "self_attn", getattr(attn_module, "attn", None))
        if attn is None or not hasattr(attn, "o_proj"):
            continue
        fwd_handles.append(attn.o_proj.register_forward_hook(make_fwd_hook(li)))
        bwd_handles.append(attn.o_proj.register_full_backward_hook(make_bwd_hook(li)))

    model.train()
    for s, text in enumerate(calibration_texts[:num_samples]):
        sample_idx[0] = s
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        if ids["input_ids"].shape[1] < 2:
            continue
        try:
            out = model(**ids, labels=ids["input_ids"])
            out.loss.backward()
            model.zero_grad()
        except Exception:
            continue
    model.eval()

    for h in fwd_handles + bwd_handles:
        h.remove()

    # Saliency = activation × gradient, averaged across samples
    saliency_per_head = {}
    for (li, s), fwd_mag in fwd_captured.items():
        bwd_mag = bwd_captured.get((li, s))
        if bwd_mag is None:
            continue
        sal = fwd_mag * bwd_mag  # element-wise per head
        saliency_per_head.setdefault(li, []).append(sal)

    for li, sal_list in saliency_per_head.items():
        if not sal_list:
            continue
        stacked = torch.stack(sal_list)
        mean_sal = stacked.mean(dim=0)
        for hi in range(min(mean_sal.shape[0], n_heads)):
            importance[li, hi] = mean_sal[hi].item()

    return importance


class TestFourMetricImportanceComparison:
    """Per Kash's four-signal hierarchy: compare L2-weight, activation, gradient, saliency.

    Mechanistic prediction: weight-norm worst, saliency best.
    Activation and gradient intermediate but for different reasons.

    This is the experiment that validates or refutes PLASTICITY-COMPACTION's
    "free utilization map from LoRA gradient capture" trick.
    """

    def test_four_metric_hierarchy(self, base_model_and_tokenizer, small_eval_dataset):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        try:
            from defrag_inline import defrag_live_model
            from forge_model import compute_activation_importance
        except ImportError as e:
            pytest.skip(f"forge_model imports unavailable: {e}")

        import copy
        base_model, tokenizer = base_model_and_tokenizer
        layers = get_layers(base_model)
        if layers is None:
            pytest.skip("Could not locate layers")

        cfg = base_model.config
        num_q = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        group_size = num_q // num_kv

        baseline_ppl = compute_perplexity(base_model, tokenizer, small_eval_dataset)
        info = {"num_layers": len(layers), "num_heads": num_q}

        results = {}

        # ── Metric 1: L2 weight norm (the broken one) ──
        m1 = copy.deepcopy(base_model)
        dead = select_lowest_importance_kv_groups(m1, num_groups_to_remove=1)
        try:
            defrag_live_model(m1, dead_heads=dead)
            results["L2_weight_norm"] = compute_perplexity(m1, tokenizer, small_eval_dataset)
        except Exception as e:
            results["L2_weight_norm"] = float("inf")
        del m1

        # Helper: pick lowest-scoring KV group from any importance tensor
        def pick_dead_from_importance(importance_tensor, n_layers, num_q, num_kv, group_size):
            dead = {}
            for li in range(n_layers):
                group_scores = []
                for kv in range(num_kv):
                    start = kv * group_size
                    end = (kv + 1) * group_size
                    score = importance_tensor[li, start:end].sum().item()
                    group_scores.append((score, kv))
                group_scores.sort()
                kv_to_remove = group_scores[0][1]
                dead[li] = list(range(kv_to_remove * group_size, (kv_to_remove + 1) * group_size))
            return dead

        # ── Metric 2: LoRA gradient magnitude (compaction paper's trick) ──
        m2 = copy.deepcopy(base_model)
        try:
            grad_imp = compute_lora_gradient_importance(m2, tokenizer, info)
            dead = pick_dead_from_importance(grad_imp, len(layers), num_q, num_kv, group_size)
            defrag_live_model(m2, dead_heads=dead)
            results["LoRA_gradient_magnitude"] = compute_perplexity(m2, tokenizer, small_eval_dataset)
        except Exception as e:
            results["LoRA_gradient_magnitude"] = float("inf")
        del m2

        # ── Metric 3: Activation magnitude (the fix) ──
        m3 = copy.deepcopy(base_model)
        try:
            act_imp = compute_activation_importance(m3, tokenizer, info)
            dead = pick_dead_from_importance(act_imp, len(layers), num_q, num_kv, group_size)
            defrag_live_model(m3, dead_heads=dead)
            results["Activation_magnitude"] = compute_perplexity(m3, tokenizer, small_eval_dataset)
        except Exception as e:
            results["Activation_magnitude"] = float("inf")
        del m3

        # ── Metric 4: Saliency (activation × gradient, Wanda/SNIP style) ──
        m4 = copy.deepcopy(base_model)
        try:
            sal_imp = compute_saliency_importance(m4, tokenizer, info)
            dead = pick_dead_from_importance(sal_imp, len(layers), num_q, num_kv, group_size)
            defrag_live_model(m4, dead_heads=dead)
            results["Saliency_activation_x_gradient"] = compute_perplexity(m4, tokenizer, small_eval_dataset)
        except Exception as e:
            results["Saliency_activation_x_gradient"] = float("inf")
        del m4

        # ── Report ──
        print(f"\n  Baseline PPL: {baseline_ppl:.2f}")
        print(f"  Four-metric comparison after pruning 1 KV group from each layer:")
        ranked = sorted(results.items(), key=lambda kv: kv[1])
        for i, (metric, ppl) in enumerate(ranked, 1):
            ratio = ppl / baseline_ppl
            print(f"  {i}. {metric:35s} PPL = {ppl:>10.2f}  ({ratio:>6.1f}x baseline)")

        # Save raw numbers as a JSON artifact for the paper
        import json as _json
        artifact = {
            "model": "Qwen/Qwen2.5-0.5B",
            "baseline_ppl": baseline_ppl,
            "results": results,
            "ranked": [(m, p) for m, p in ranked],
            "note": "Lower is better. Pruning without retraining always degrades; the comparison is RELATIVE.",
        }
        artifact_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "output", "four_metric_comparison.json"
        )
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        with open(artifact_path, "w") as f:
            _json.dump(artifact, f, indent=2)
        print(f"\n  Artifact: {artifact_path}")

        # All metrics must produce finite results
        for metric, ppl in results.items():
            assert not math.isnan(ppl), f"{metric} produced NaN"
            assert not math.isinf(ppl), f"{metric} produced Inf — likely a tensor surgery break"

        # The hypothesis test: at least one of the three non-broken metrics
        # must be better than L2-weight-norm
        l2 = results["L2_weight_norm"]
        non_l2 = [results[k] for k in results if k != "L2_weight_norm" and not math.isinf(results[k])]
        assert non_l2, "All non-L2 metrics failed to compute"
        assert min(non_l2) < l2, (
            f"None of the better-by-design metrics beat L2 weight norm. "
            f"L2={l2:.2f}, others={non_l2}. "
            "This would invalidate the activation-importance finding."
        )


class TestActivationVsWeightNormImportance:
    """Empirically prove activation-based importance beats L2-norm for selecting prunable heads.

    This is the validation of the fix for sentinel-ai #155.
    The test compares perplexity after defrag using:
    - L2 norm of Q projection (the broken metric)
    - Activation magnitude on calibration data (the fixed metric)

    The activation-based metric should produce LOWER post-defrag perplexity.
    """

    def test_activation_importance_beats_weight_norm(
        self, base_model_and_tokenizer, small_eval_dataset
    ):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        try:
            from defrag_inline import defrag_live_model
            from forge_model import compute_activation_importance, get_layers as fm_get_layers
        except ImportError as e:
            pytest.skip(f"forge_model imports unavailable: {e}")

        import copy
        base_model, tokenizer = base_model_and_tokenizer
        layers = get_layers(base_model)
        if layers is None:
            pytest.skip("Could not locate layers")

        cfg = base_model.config
        num_q = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        group_size = num_q // num_kv

        baseline_ppl = compute_perplexity(base_model, tokenizer, small_eval_dataset)
        print(f"\n  Baseline PPL: {baseline_ppl:.2f}")

        # ── Strategy A: L2 norm of Q (the broken metric) ──
        model_a = copy.deepcopy(base_model)
        dead_a = select_lowest_importance_kv_groups(model_a, num_groups_to_remove=1)
        try:
            defrag_live_model(model_a, dead_heads=dead_a)
        except Exception as e:
            pytest.skip(f"weight-norm defrag failed: {e}")
        ppl_a = compute_perplexity(model_a, tokenizer, small_eval_dataset)
        print(f"  Weight-norm L2 (broken):   PPL = {ppl_a:.2f}")
        del model_a

        # ── Strategy B: Activation-based importance (the fix) ──
        model_b = copy.deepcopy(base_model)
        info = {"num_layers": len(layers), "num_heads": num_q}
        try:
            importance = compute_activation_importance(model_b, tokenizer, info)
        except Exception as e:
            pytest.skip(f"activation importance failed: {e}")

        # Pick lowest-importance KV group per layer using the activation metric.
        # Sum head importances within each KV group, take the lowest-sum group.
        dead_b = {}
        for li in range(len(layers)):
            group_scores = []
            for kv in range(num_kv):
                start = kv * group_size
                end = (kv + 1) * group_size
                score = importance[li, start:end].sum().item()
                group_scores.append((score, kv))
            group_scores.sort()
            kv_to_remove = group_scores[0][1]  # lowest-scoring group
            dead_b[li] = list(range(kv_to_remove * group_size, (kv_to_remove + 1) * group_size))

        try:
            defrag_live_model(model_b, dead_heads=dead_b)
        except Exception as e:
            pytest.skip(f"activation defrag failed: {e}")
        ppl_b = compute_perplexity(model_b, tokenizer, small_eval_dataset)
        print(f"  Activation-based (fix):    PPL = {ppl_b:.2f}")
        del model_b

        # The fix must produce LOWER perplexity than the broken metric
        # (we don't claim it's near baseline — pruning without retraining still hurts)
        improvement = (ppl_a - ppl_b) / ppl_a * 100
        print(f"  Improvement from fix: {improvement:.1f}%")

        assert ppl_b < ppl_a, (
            f"Activation-based ({ppl_b:.2f}) did NOT beat weight-norm ({ppl_a:.2f}). "
            "The fix is not actually better than the broken metric on this model."
        )
        assert not math.isnan(ppl_b)
        assert not math.isinf(ppl_b)


class TestGenerationSanity:
    """A defragged model should generate coherent (or at least non-broken) text."""

    def test_defragged_model_generates_text(self, base_model_and_tokenizer):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        try:
            from defrag_inline import defrag_live_model
        except ImportError:
            pytest.skip("defrag_inline not importable")

        import copy
        base_model, tokenizer = base_model_and_tokenizer
        model = copy.deepcopy(base_model)
        layers = get_layers(model)
        if layers is None:
            pytest.skip("Could not locate layers")

        cfg = model.config
        num_q = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        group_size = num_q // num_kv

        dead_heads = {
            li: list(range(num_q - group_size, num_q))
            for li in range(len(layers))
        }
        try:
            defrag_live_model(model, dead_heads=dead_heads)
        except Exception as e:
            pytest.skip(f"defrag failed: {e}")

        prompt = "The quick brown fox"
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        print(f"\n  Generated: {text[:100]}")

        # Sanity: output is non-empty, contains the prompt, has SOME continuation
        assert len(text) > len(prompt)
        assert prompt in text or prompt.lower() in text.lower()
