"""Synthetic test harness for cpu_expert_prune_v2.py.

Creates a tiny MoE in the qwen3_moe layout, runs the v2 expert pruning
script against it, and asserts:
- The router gate rows are correctly sliced
- The dropped-expert MLP tensors are gone
- The surviving-expert MLP tensors are renamed sequentially
- The non-MoE tensors (embeddings, attention, norms) are passed through unchanged
- The config has num_experts updated
- The metadata sidecar has the right structure

If this passes, the v2 script is safe to run on the real Qwen3-Coder-30B-A3B.

Run:
    python -m scripts.test_cpu_expert_prune_v2
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# ── Synthetic config ────────────────────────────────────────────────────────
HIDDEN = 64
MOE_INTER = 32
NUM_LAYERS = 4
NUM_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2
KEEP_EXPERTS = 4  # prune from 8 → 4

# Importance pattern: in each layer, give experts at FIXED indices [1, 3, 5, 7]
# the highest router-gate norm. The other 4 experts get tiny norms. After
# pruning, the surviving experts should be exactly [1, 3, 5, 7] in every
# layer, renumbered to [0, 1, 2, 3] in the output.
IMPORTANT_INDICES = [1, 3, 5, 7]
def _make_tiny_moe(model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN * 4,
        "moe_intermediate_size": MOE_INTER,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))

    state: dict[str, torch.Tensor] = {}

    # Embeddings + lm_head + final norm (all non-MoE, must pass through unchanged)
    state["model.embed_tokens.weight"] = torch.randn(128, HIDDEN, dtype=torch.float16)
    state["lm_head.weight"] = torch.randn(128, HIDDEN, dtype=torch.float16)
    state["model.norm.weight"] = torch.randn(HIDDEN, dtype=torch.float16)

    for L in range(NUM_LAYERS):
        # Attention (non-MoE, must pass through unchanged)
        state[f"model.layers.{L}.self_attn.q_proj.weight"] = torch.randn(64, HIDDEN, dtype=torch.float16)
        state[f"model.layers.{L}.self_attn.k_proj.weight"] = torch.randn(32, HIDDEN, dtype=torch.float16)
        state[f"model.layers.{L}.self_attn.v_proj.weight"] = torch.randn(32, HIDDEN, dtype=torch.float16)
        state[f"model.layers.{L}.self_attn.o_proj.weight"] = torch.randn(HIDDEN, 64, dtype=torch.float16)
        state[f"model.layers.{L}.input_layernorm.weight"] = torch.randn(HIDDEN, dtype=torch.float16)
        state[f"model.layers.{L}.post_attention_layernorm.weight"] = torch.randn(HIDDEN, dtype=torch.float16)

        # Router gate: rows for "important" experts (fixed indices) get large
        # norm, others tiny. After pruning, surviving == IMPORTANT_INDICES.
        gate = torch.zeros(NUM_EXPERTS, HIDDEN, dtype=torch.float16)
        for E in range(NUM_EXPERTS):
            if E in IMPORTANT_INDICES:
                gate[E] = torch.randn(HIDDEN, dtype=torch.float16) * 10.0
            else:
                gate[E] = torch.randn(HIDDEN, dtype=torch.float16) * 0.01
        state[f"model.layers.{L}.mlp.gate.weight"] = gate

        # Per-expert MLPs (unfused layout). Mark each expert's tensors with a
        # signature value so we can verify the renaming preserved the right ones.
        for E in range(NUM_EXPERTS):
            sig = float(L * 100 + E)  # unique per (layer, expert)
            state[f"model.layers.{L}.mlp.experts.{E}.gate_proj.weight"] = torch.full(
                (MOE_INTER, HIDDEN), sig, dtype=torch.float16
            )
            state[f"model.layers.{L}.mlp.experts.{E}.up_proj.weight"] = torch.full(
                (MOE_INTER, HIDDEN), sig + 0.5, dtype=torch.float16
            )
            state[f"model.layers.{L}.mlp.experts.{E}.down_proj.weight"] = torch.full(
                (HIDDEN, MOE_INTER), -sig, dtype=torch.float16
            )

    save_file(state, str(model_dir / "model.safetensors"))
    # weight_map index
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
        "weight_map": {k: "model.safetensors" for k in state.keys()},
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return state


def _read_all(model_dir: Path) -> dict[str, torch.Tensor]:
    out = {}
    for sp in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def main():
    print("=" * 60)
    print("test_cpu_expert_prune_v2")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        out = tmp / "out"

        # Build synthetic source
        src_state = _make_tiny_moe(src)
        print(f"\n[1] built synthetic MoE: {NUM_LAYERS} layers × {NUM_EXPERTS} experts")
        print(f"    expected surviving experts per layer: 2L..2L+3 (the 4 high-norm experts)")
        print(f"    source tensors: {len(src_state)}")

        # Run v2 script
        script_path = Path(__file__).resolve().parent / "cpu_expert_prune_v2.py"
        print(f"\n[2] running cpu_expert_prune_v2.py --keep-experts {KEEP_EXPERTS}")
        result = subprocess.run(
            [sys.executable, str(script_path), str(src), str(out),
             "--keep-experts", str(KEEP_EXPERTS)],
            capture_output=True, text=True,
        )
        print("    stdout:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.returncode != 0:
            print("    stderr:", result.stderr)
            sys.exit(f"FAIL: script exited {result.returncode}")

        # Load output
        out_state = _read_all(out)
        print(f"\n[3] verifying output ({len(out_state)} tensors)")

        # Read sidecar
        sidecar = json.loads((out / "expert_prune.metadata.v1.json").read_text())
        print(f"    sidecar tensors: kept={sidecar['tensors']['kept_unchanged']} "
              f"renamed={sidecar['tensors']['kept_renamed']} "
              f"router_sliced={sidecar['tensors']['router_gate_sliced']} "
              f"dropped={sidecar['tensors']['dropped_expert']}")

        errors: list[str] = []

        # ── Check 1: non-MoE tensors are passed through unchanged
        for k in ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]:
            if k not in out_state:
                errors.append(f"missing pass-through tensor {k}")
                continue
            if not torch.equal(out_state[k], src_state[k]):
                errors.append(f"pass-through tensor {k} was modified")

        for L in range(NUM_LAYERS):
            for sub in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                        "self_attn.o_proj", "input_layernorm", "post_attention_layernorm"):
                k = f"model.layers.{L}.{sub}.weight"
                if k not in out_state:
                    errors.append(f"missing pass-through tensor {k}")
                    continue
                if not torch.equal(out_state[k], src_state[k]):
                    errors.append(f"pass-through tensor {k} was modified")

        # ── Check 2: router gate is sliced to KEEP_EXPERTS rows, with the right rows
        for L in range(NUM_LAYERS):
            k = f"model.layers.{L}.mlp.gate.weight"
            if k not in out_state:
                errors.append(f"missing router gate {k}")
                continue
            new_gate = out_state[k]
            if new_gate.shape != (KEEP_EXPERTS, HIDDEN):
                errors.append(f"router gate {k} has wrong shape: {tuple(new_gate.shape)} != ({KEEP_EXPERTS}, {HIDDEN})")
                continue
            # The selected experts should be exactly IMPORTANT_INDICES, sorted
            for new_idx, old_idx in enumerate(IMPORTANT_INDICES):
                if not torch.equal(new_gate[new_idx], src_state[k][old_idx]):
                    errors.append(f"router gate {k} row {new_idx} != source row {old_idx}")

        # ── Check 3: surviving expert tensors are renamed and have the right signatures
        for L in range(NUM_LAYERS):
            for new_idx, old_idx in enumerate(IMPORTANT_INDICES):
                sig = float(L * 100 + old_idx)
                # Each surviving expert has 3 tensors (gate, up, down) at the new index
                gate_k = f"model.layers.{L}.mlp.experts.{new_idx}.gate_proj.weight"
                up_k = f"model.layers.{L}.mlp.experts.{new_idx}.up_proj.weight"
                down_k = f"model.layers.{L}.mlp.experts.{new_idx}.down_proj.weight"
                for k, expected_val in [(gate_k, sig), (up_k, sig + 0.5), (down_k, -sig)]:
                    if k not in out_state:
                        errors.append(f"missing renamed expert tensor {k}")
                        continue
                    actual = out_state[k][0, 0].item()
                    if abs(actual - expected_val) > 0.5:  # fp16 tolerance
                        errors.append(f"expert tensor {k} has signature {actual} != expected {expected_val}")

        # ── Check 4: dropped expert tensors are NOT in the output
        for L in range(NUM_LAYERS):
            dropped_old_indices = [E for E in range(NUM_EXPERTS) if E not in IMPORTANT_INDICES]
            for E in dropped_old_indices:
                # Old name should not appear in output (the new naming is sequential 0..3)
                # In particular, indices 4..7 should not appear in output for any layer
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    old_k = f"model.layers.{L}.mlp.experts.{E}.{proj}.weight"
                    if old_k in out_state and E >= KEEP_EXPERTS:
                        errors.append(f"dropped expert tensor {old_k} present in output")
            # No tensor with index >= KEEP_EXPERTS should exist
            for E in range(KEEP_EXPERTS, NUM_EXPERTS):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    old_k = f"model.layers.{L}.mlp.experts.{E}.{proj}.weight"
                    if old_k in out_state:
                        errors.append(f"out-of-range expert index in output: {old_k}")

        # ── Check 5: config has num_experts updated
        out_cfg = json.loads((out / "config.json").read_text())
        if out_cfg["num_experts"] != KEEP_EXPERTS:
            errors.append(f"config num_experts = {out_cfg['num_experts']} != {KEEP_EXPERTS}")

        # ── Check 6: weight_map index
        idx = json.loads((out / "model.safetensors.index.json").read_text())
        wm = idx["weight_map"]
        for k in out_state:
            if k not in wm:
                errors.append(f"weight_map missing tensor {k}")

        # ── Report
        print(f"\n[4] {len(errors)} errors")
        if errors:
            for e in errors[:20]:
                print(f"  ✗ {e}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")
            sys.exit(1)

        # Sidecar tensor count sanity
        # Per layer: 1 router gate sliced + 4 kept_renamed (gate, up, down) × 4 = 12 renamed
        # Pass-through per layer: 6 (q,k,v,o,2 norms)
        # Plus 3 global pass-through (embed, lm_head, final norm)
        expected_router_sliced = NUM_LAYERS  # 4
        expected_renamed = NUM_LAYERS * KEEP_EXPERTS * 3  # 4 * 4 * 3 = 48
        expected_dropped = NUM_LAYERS * (NUM_EXPERTS - KEEP_EXPERTS) * 3  # 4 * 4 * 3 = 48
        expected_unchanged = 3 + NUM_LAYERS * 6  # 3 + 24 = 27

        if sidecar["tensors"]["router_gate_sliced"] != expected_router_sliced:
            print(f"  WARN: router_gate_sliced = {sidecar['tensors']['router_gate_sliced']} expected {expected_router_sliced}")
        if sidecar["tensors"]["kept_renamed"] != expected_renamed:
            print(f"  WARN: kept_renamed = {sidecar['tensors']['kept_renamed']} expected {expected_renamed}")
        if sidecar["tensors"]["dropped_expert"] != expected_dropped:
            print(f"  WARN: dropped_expert = {sidecar['tensors']['dropped_expert']} expected {expected_dropped}")
        if sidecar["tensors"]["kept_unchanged"] != expected_unchanged:
            print(f"  WARN: kept_unchanged = {sidecar['tensors']['kept_unchanged']} expected {expected_unchanged}")

        print("\n  ✓ ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
