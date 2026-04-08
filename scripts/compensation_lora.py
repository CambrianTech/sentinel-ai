"""
compensation_lora.py — compensation LoRA via teacher distillation.

Trains a LoRA adapter on a *pruned* student model to recover capability lost
to head pruning, by minimizing a distillation loss against the *unpruned*
teacher model. Implements the architectural pattern from
``models/unet_transformer.py``'s ``BaselineIntegratedBlock`` (baseline
integration via learnable adapter + gate) as a model-agnostic LoRA wrapper
that works on any HuggingFace transformers model.

This addresses the §4.1.3.2 PPL/HumanEval disconnect at the *structural*
level rather than the calibration-data level: the LoRA learns to add a
correction that brings the student's hidden states closer to the teacher's
at each layer, recovering held-out task generalization that the surviving
heads alone cannot absorb via standard task-loss fine-tuning.

The architectural pattern, mapped from ``BaselineIntegratedBlock``:

- ``baseline_adapter`` (Linear, embed_dim → embed_dim, learnable)
  → LoRA's low-rank adapter on attention/FFN projections (also learnable
  Linear, low rank, applied to the same embed_dim → embed_dim path)
- ``baseline_gate`` (learnable scalar, sigmoid-gated fusion factor)
  → LoRA's ``alpha / rank`` scaling factor (effectively a fixed gate, but
  the rank itself is learnable in the sense that increasing rank lets the
  LoRA carry more compensation information)
- ``ln_baseline`` (LayerNorm before adapter projection)
  → handled implicitly by the LoRA adapter living inside the existing
  layer norms of the student model

The training objective is a *distillation* loss (teacher hidden state vs.
student hidden state, MSE per layer; or KL divergence on output logits)
rather than a task loss. This is what makes the LoRA recover the pruned
heads' contributions instead of just fine-tuning the student for next-token
prediction on a fixed dataset (which is what the v2-7B forge does today
and what produces the 54.9 / 48.8 result that §4.1.3.2 explains as
calibration-distribution-narrow).

Usage::

    python compensation_lora.py \\
        --teacher /path/to/qwen2.5-coder-7b-base \\
        --student /path/to/v2-7b-pruned \\
        --calibration-data /path/to/heldout_mix.jsonl \\
        --output /path/to/v2-7b-compensated \\
        --steps 500 \\
        --loss-type mse_hidden

The calibration data is the load-bearing input. It should be a JSONL file
with ``{"text": "..."}`` entries drawn from the *held-out* task distribution
the lab cares about — HumanEval problems for code, GSM8K for math, MMLU for
knowledge, or a mixture spanning the deployment surface. The §4.1.3.2 fix
specifically requires the calibration data to be held-out from the student's
fine-tuning corpus; otherwise the same calibration-distribution narrowness
that produced the disconnect will recur in the compensation step.

The script is model-agnostic. It works for any HuggingFace transformers
``CausalLM`` model whose ``hidden_size`` and ``num_hidden_layers`` match
between teacher and student (which is guaranteed by pad-mode defrag, since
that operation preserves both). It does not modify any existing forge
pipeline files; it is a pure addition to ``scripts/`` and runs as a separate
post-forge stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Heavy ML imports are LAZY (inside the functions that use them) so this
# module is importable on machines without torch / transformers / peft
# installed — the family-adapter Tier 1 dispatch path imports this module
# without ever loading models, and so does the unit-test layer that
# verifies the importable contract. The CLI path and the actual
# distillation work both still need the heavy deps and will fail loudly
# at the lazy import site if they're missing.

# Same-directory import; the harness_checks module enforces preconditions
# at the contract layer (used by both _compensate_inner and the CLI wrapper).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_checks import assert_explicit_head_dim  # noqa: E402

# Canonical loss-type set. Keep in sync with the CLI choices and the
# adapter contract test (test_compensation_lora_api.test_compensate_lora_raises_on_invalid_loss_type).
VALID_LOSS_TYPES = frozenset({"mse_hidden", "kl_logits", "both"})
VALID_TEACHER_QUANTS = frozenset({"8bit", "4bit"})
VALID_STUDENT_QUANTS = frozenset({"fp16", "4bit"})


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _bnb_config(quant: str):
    """Build a BitsAndBytesConfig for the named quant tier, or None for fp16."""
    import torch
    from transformers import BitsAndBytesConfig
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return None


def load_teacher(path: str, device: str, quant: str = "8bit"):
    """Load the unpruned teacher, frozen.

    quant ∈ {"8bit","4bit"}. Teacher is forward-pass-only (no gradients) so
    quantization is essentially free for distillation logits. For 7B-class
    teachers 8bit fits comfortably alongside an fp16 student on a 32 GB GPU.
    For 30B-class teachers 4bit is required because the student also has to
    fit in 4bit + LoRA + activations + KV cache simultaneously.
    """
    from transformers import AutoModelForCausalLM
    bnb_config = _bnb_config(quant)
    if bnb_config is None:
        raise ValueError(f"unknown teacher quant: {quant!r} (use 8bit or 4bit)")
    teacher = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=bnb_config,
        device_map=device,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def load_student(path: str, device: str, quant: str = "fp16"):
    """Load the pruned student, with gradient checkpointing enabled.

    quant ∈ {"fp16","4bit"}. fp16 is the default for ≤14B-class students;
    backprop runs through the full-precision weights and the LoRA is fp16.

    For 30B-class students, fp16 (~40 GB) doesn't fit a 32 GB GPU at all,
    so 4bit (NF4 + double quant) is required. In 4bit mode the student is
    QLoRA-style: base weights are frozen 4bit, LoRA adapters are fp16, and
    only the LoRA parameters are trained. `prepare_model_for_kbit_training`
    handles the dtype dance and the gradient checkpointing setup.
    """
    import torch
    from transformers import AutoModelForCausalLM
    if quant == "fp16":
        student = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map=device,
        )
    elif quant == "4bit":
        from peft import prepare_model_for_kbit_training
        bnb_config = _bnb_config("4bit")
        student = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=bnb_config,
            device_map=device,
        )
        student = prepare_model_for_kbit_training(
            student, use_gradient_checkpointing=True
        )
    else:
        raise ValueError(f"unknown student quant: {quant!r} (use fp16 or 4bit)")
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        # Some HF model classes need this flag flipped for gradient checkpointing
        # to interact correctly with PEFT-wrapped models.
        if hasattr(student.config, "use_cache"):
            student.config.use_cache = False
    return student


# ---------------------------------------------------------------------------
# LoRA attachment
# ---------------------------------------------------------------------------


def attach_compensation_lora(
    student,
    rank: int,
    alpha: int,
    target_modules: list[str],
    dropout: float = 0.05,
):
    """Wrap the student with a compensation LoRA targeting attention + FFN projections.

    The LoRA's role is structurally analogous to ``BaselineIntegratedBlock``'s
    ``baseline_adapter`` — a low-rank learnable correction that adapts the
    student's representation to be closer to the teacher's. The LoRA's
    ``alpha / rank`` scaling factor plays the role of the fixed
    ``baseline_gate``; making the rank itself a hyperparameter lets the
    compensation LoRA carry more or less correction information depending on
    how much capacity was lost to pruning.

    Default target modules cover both attention and FFN projections so the
    compensation can route through whichever path the student's pruned heads
    were contributing to. Excluding attention projections from the target list
    is a valid alternative if the user wants compensation that doesn't touch
    the attention layout at all (purely FFN-side correction); see the
    ``--target-modules`` CLI flag.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(student, lora_config)


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------


def compute_distillation_loss(
    teacher_hidden,
    student_hidden,
    teacher_logits,
    student_logits,
    loss_type: str,
    temperature: float = 2.0,
):
    """Compute distillation loss between teacher and student.

    Three loss types are supported:

    ``mse_hidden``
        MSE on per-layer hidden states. Matches the ``BaselineIntegratedBlock``
        pattern of "learn a residual correction at each layer that brings the
        student's hidden state closer to the teacher's." This is the most
        directly U-Net-analogous loss and the default for v0.

    ``kl_logits``
        KL divergence on output logits with temperature smoothing. Standard
        knowledge distillation as in Hinton et al. 2015. Less directly tied
        to the U-Net pattern but is the canonical distillation objective and
        is a useful comparison row.

    ``both``
        Sum of MSE-hidden and KL-logits, equally weighted. The combined loss
        targets both intermediate-layer alignment and output-distribution
        alignment simultaneously, which empirically tends to be more robust
        than either alone in the knowledge-distillation literature.
    """
    import torch.nn.functional as F
    losses = []

    if loss_type in ("mse_hidden", "both"):
        if len(teacher_hidden) != len(student_hidden):
            raise AssertionError(
                f"teacher has {len(teacher_hidden)} hidden states, student has "
                f"{len(student_hidden)}; both models must have the same number "
                f"of layers for hidden-state distillation. If they don't, the "
                f"student is from a different model family than the teacher and "
                f"this script cannot bridge them — use a teacher from the same "
                f"family as the student's pre-prune base."
            )
        layer_losses = []
        for t, s in zip(teacher_hidden, student_hidden):
            if t.shape != s.shape:
                raise AssertionError(
                    f"teacher hidden state shape {tuple(t.shape)} != student "
                    f"shape {tuple(s.shape)}; hidden_size mismatch. Pad-mode "
                    f"defrag preserves hidden_size, so if you are seeing this "
                    f"assertion the student was pruned with slice mode (which "
                    f"shrinks hidden_size) and is not compatible with the "
                    f"teacher's hidden_size. Use a pad-mode student or build a "
                    f"projection adapter (not yet implemented)."
                )
            # Cast to float32 for the loss to avoid fp16 underflow on small
            # residual differences early in training.
            layer_losses.append(F.mse_loss(s.float(), t.float()))
        losses.append(sum(layer_losses) / len(layer_losses))

    if loss_type in ("kl_logits", "both"):
        T = temperature
        teacher_probs = F.softmax(teacher_logits.float() / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits.float() / T, dim=-1)
        kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (T * T)
        losses.append(kl)

    if not losses:
        raise AssertionError(
            f"unknown loss_type: {loss_type!r}; expected one of "
            f"{{'mse_hidden', 'kl_logits', 'both'}}"
        )

    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------


def make_jsonl_text_dataset(path: str, tokenizer, max_length: int = 1024):
    """Construct a calibration dataset from a JSONL file with ``{"text": "..."}`` entries.

    Returns a torch.utils.data.Dataset subclass instance. The class is
    defined inside this factory so the torch import is lazy — the module
    can be imported on a Mac without torch installed for the unit-test layer
    that verifies the importable contract; the class only gets constructed
    when an actual distillation run calls this factory.

    The caller is responsible for constructing the JSONL with held-out task
    examples (HumanEval problems, GSM8K, MMLU, or a mixture). This is the
    held-out-aware calibration data per §4.1.3.2's structural fix; it must
    not be drawn from the student's fine-tuning corpus or the calibration-
    distribution narrowness will recur.
    """
    from torch.utils.data import Dataset

    class JsonlTextDataset(Dataset):
        def __init__(self, path: str, tokenizer, max_length: int):
            text_examples: list[str] = []
            with open(path) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if "text" not in record:
                        raise AssertionError(
                            f"calibration data line in {path} is missing the "
                            f"'text' key; expected JSONL with at least "
                            f"{{'text': '...'}} per line"
                        )
                    text_examples.append(record["text"])
            if not text_examples:
                raise AssertionError(
                    f"calibration data file {path} is empty; provide at least one "
                    f"held-out task example for the compensation LoRA to train against"
                )
            self.examples = text_examples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, idx: int) -> dict:
            return self.tokenizer(
                self.examples[idx],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

    return JsonlTextDataset(path, tokenizer, max_length)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compensation LoRA via teacher distillation."
    )
    parser.add_argument(
        "--teacher",
        required=True,
        help="path or HF id of the unpruned teacher model",
    )
    parser.add_argument(
        "--student",
        required=True,
        help="path of the pruned student model (from forge_v2_pipeline.sh)",
    )
    parser.add_argument(
        "--calibration-data",
        required=True,
        help=(
            "JSONL file of held-out task texts for the distillation calibration "
            "(must not overlap the student's fine-tuning corpus per §4.1.3.2)"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output directory for the compensated student model",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--loss-type",
        choices=["mse_hidden", "kl_logits", "both"],
        default="mse_hidden",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        help="LoRA target modules; default covers attention + FFN projections",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="print loss every N steps",
    )
    parser.add_argument(
        "--teacher-quant",
        choices=["8bit", "4bit"],
        default="8bit",
        help=(
            "Teacher quantization. Default 8bit fits 7-14B teachers alongside "
            "an fp16 student on a 32 GB GPU. For 30B+ teachers use 4bit "
            "(required when student is also 4bit, the QLoRA pattern)."
        ),
    )
    parser.add_argument(
        "--student-quant",
        choices=["fp16", "4bit"],
        default="fp16",
        help=(
            "Student quantization. Default fp16 for ≤14B students. For 30B+ "
            "students use 4bit (NF4 + double quant + QLoRA pattern); base "
            "weights are frozen 4bit and only the LoRA adapters are trained."
        ),
    )
    return parser.parse_args()


def _validate_compensate_inputs(
    *,
    calibration_data: Path,
    output: Path,
    loss_type: str,
    teacher_quant: str,
    target_modules: list[str],
    steps: int,
    lora_rank: int,
    lora_alpha: int,
) -> None:
    """Validate every input parameter at the entry surface, before touching
    any heavy machinery. Loud failures here mean the contract is wrong; the
    error messages name the offending field so the caller can fix it.

    Called by both compensate_lora and compensate_lora_from_paths so the
    validation is consistent across entry points and the unit tests can
    exercise it without loading any models.
    """
    if loss_type not in VALID_LOSS_TYPES:
        raise ValueError(
            f"loss_type must be one of {sorted(VALID_LOSS_TYPES)}, got {loss_type!r}"
        )
    if teacher_quant not in VALID_TEACHER_QUANTS:
        raise ValueError(
            f"teacher_quant must be one of {sorted(VALID_TEACHER_QUANTS)}, "
            f"got {teacher_quant!r}"
        )
    if not isinstance(target_modules, (list, tuple)) or not target_modules:
        raise ValueError(
            f"target_modules must be a non-empty list of LoRA target module "
            f"names, got {target_modules!r}"
        )
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if lora_rank < 1:
        raise ValueError(f"lora_rank must be >= 1, got {lora_rank}")
    if lora_alpha < 1:
        raise ValueError(f"lora_alpha must be >= 1, got {lora_alpha}")
    calibration_data = Path(calibration_data)
    if not calibration_data.exists():
        raise FileNotFoundError(
            f"calibration_data path {calibration_data} does not exist. The "
            f"§4.1.3.4.1 discipline gate requires the calibration corpus to be "
            f"present and hash-pinned before the compensation LoRA stage runs."
        )


def _compensate_inner(
    *,
    teacher,
    student,
    tokenizer,
    device: str,
    calibration_data: Path,
    output: Path,
    steps: int,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float,
    loss_type: str,
    target_modules: list[str],
    max_length: int,
    log_every: int,
) -> dict[str, Any]:
    """The actual distillation training loop. Caller provides loaded teacher
    + loaded student + tokenizer; this function attaches the LoRA, runs the
    training loop, merges the LoRA into the student weights, saves the
    compensated student to `output`, and returns a metadata dict.

    Both compensate_lora and compensate_lora_from_paths wrap this function.
    """
    import torch
    from transformers import AutoConfig

    # Preconditions: the STUDENT must carry head_dim explicitly (post-defrag
    # invariant from §4.1.3 / Finding 6 — the v1 bug we explicitly defend
    # against). The TEACHER does NOT need this assertion: an unmodified base
    # model legitimately has implicit head_dim because no defrag happened.
    print("[compensation_lora] checking preconditions")
    assert_explicit_head_dim(student.config)
    if teacher.config.hidden_size != student.config.hidden_size:
        raise AssertionError(
            f"teacher hidden_size={teacher.config.hidden_size} != "
            f"student hidden_size={student.config.hidden_size}; the student "
            f"must have been pruned with pad-mode defrag (which preserves "
            f"hidden_size). Slice-mode pruning shrinks hidden_size and is not "
            f"compatible with this distillation script."
        )
    if teacher.config.num_hidden_layers != student.config.num_hidden_layers:
        raise AssertionError(
            f"teacher num_hidden_layers={teacher.config.num_hidden_layers} != "
            f"student num_hidden_layers={student.config.num_hidden_layers}; "
            f"layer count must match for hidden-state distillation"
        )

    print(
        f"[compensation_lora] attaching compensation LoRA "
        f"(r={lora_rank}, alpha={lora_alpha}, target_modules={target_modules})"
    )
    student = attach_compensation_lora(
        student=student,
        rank=lora_rank,
        alpha=lora_alpha,
        target_modules=target_modules,
    )
    student.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[compensation_lora] loading calibration data from {calibration_data}")
    dataset = make_jsonl_text_dataset(
        path=str(calibration_data),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    print(f"[compensation_lora] loaded {len(dataset)} calibration examples")

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    print(f"[compensation_lora] training for {steps} steps with loss={loss_type}")
    student.train()
    step = 0
    epoch = 0
    final_loss = None
    while step < steps:
        epoch += 1
        for example in dataset:
            if step >= steps:
                break

            input_ids = example["input_ids"].to(device).squeeze(0)
            attention_mask = example["attention_mask"].to(device).squeeze(0)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            student_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            loss = compute_distillation_loss(
                teacher_hidden=teacher_out.hidden_states,
                student_hidden=student_out.hidden_states,
                teacher_logits=teacher_out.logits,
                student_logits=student_out.logits,
                loss_type=loss_type,
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            final_loss = loss.item()

            step += 1
            if step == 1 or step % log_every == 0 or step == steps:
                print(
                    f"[compensation_lora] epoch {epoch} step {step}/{steps} "
                    f"loss={final_loss:.6f}"
                )

    print("[compensation_lora] training complete; merging LoRA into student weights")
    student = student.merge_and_unload()

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compensation_lora] saving compensated student to {out_dir}")
    student.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    # Verify the saved config still carries head_dim explicitly.
    saved_config = AutoConfig.from_pretrained(out_dir)
    assert_explicit_head_dim(saved_config)

    print(f"[compensation_lora] done. compensated model at {out_dir}")

    return {
        "output_dir": str(out_dir),
        "steps_completed": step,
        "final_loss": final_loss,
        "loss_type": loss_type,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(target_modules),
    }


# ── Importable API ──────────────────────────────────────────────────────────


def compensate_lora(
    *,
    student,
    student_tokenizer,
    teacher_path: str,
    teacher_quant: str,
    calibration_data: str | Path,
    output: str | Path,
    steps: int,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float,
    loss_type: str,
    target_modules: list[str],
    max_length: int,
    log_every: int = 25,
) -> dict[str, Any]:
    """§ 4.1.3.3 KL-distillation-against-teacher compensation LoRA.

    Adapter-side entry point: caller provides an ALREADY-LOADED student
    model + tokenizer (the alloy_executor has already loaded them), and
    this function loads the teacher itself in the requested quant tier
    (8bit / 4bit), runs the distillation training loop, merges the LoRA,
    saves the compensated student, and returns a metadata dict.

    Used by QwenDenseBase._train_compensation in scripts/adapters/qwen_dense_base.py.

    Validates every input at the entry surface BEFORE loading the teacher
    or touching the student, so contract violations fail fast with clear
    error messages.
    """
    calibration_data = Path(calibration_data)
    output = Path(output)

    _validate_compensate_inputs(
        calibration_data=calibration_data,
        output=output,
        loss_type=loss_type,
        teacher_quant=teacher_quant,
        target_modules=target_modules,
        steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    # Determine the device the student is on. The student is already loaded
    # so its device is fixed; the teacher gets loaded onto the same device.
    import torch
    if hasattr(student, "device"):
        device = str(student.device)
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"[compensation_lora] device={device}")
    print(f"[compensation_lora] loading teacher from {teacher_path} (bnb-{teacher_quant}, frozen)")
    teacher = load_teacher(teacher_path, device, quant=teacher_quant)

    return _compensate_inner(
        teacher=teacher,
        student=student,
        tokenizer=student_tokenizer,
        device=device,
        calibration_data=calibration_data,
        output=output,
        steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        loss_type=loss_type,
        target_modules=list(target_modules),
        max_length=max_length,
        log_every=log_every,
    )


def compensate_lora_from_paths(
    *,
    teacher_path: str,
    student_path: str,
    student_quant: str,
    calibration_data: str | Path,
    output: str | Path,
    steps: int,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float,
    loss_type: str,
    target_modules: list[str],
    max_length: int,
    teacher_quant: str = "8bit",
    log_every: int = 25,
) -> dict[str, Any]:
    """§ 4.1.3.3 compensation LoRA — CLI entry point.

    Loads BOTH teacher (in teacher_quant) and student (in student_quant)
    from disk paths, then delegates to _compensate_inner. Used by main()
    and any caller that wants this script to handle all model loading.

    Same param contract as compensate_lora() but with student_path / student_quant
    in place of pre-loaded student object.
    """
    calibration_data = Path(calibration_data)
    output = Path(output)

    _validate_compensate_inputs(
        calibration_data=calibration_data,
        output=output,
        loss_type=loss_type,
        teacher_quant=teacher_quant,
        target_modules=target_modules,
        steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    if student_quant not in VALID_STUDENT_QUANTS:
        raise ValueError(
            f"student_quant must be one of {sorted(VALID_STUDENT_QUANTS)}, "
            f"got {student_quant!r}"
        )

    import torch
    from transformers import AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[compensation_lora] device={device}")
    print(f"[compensation_lora] loading teacher from {teacher_path} (bnb-{teacher_quant}, frozen)")
    teacher = load_teacher(teacher_path, device, quant=teacher_quant)

    print(f"[compensation_lora] loading student from {student_path} ({student_quant}, grad-checkpointed)")
    student = load_student(student_path, device, quant=student_quant)

    print(f"[compensation_lora] loading tokenizer from {teacher_path}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_path)

    return _compensate_inner(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        device=device,
        calibration_data=calibration_data,
        output=output,
        steps=steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        loss_type=loss_type,
        target_modules=list(target_modules),
        max_length=max_length,
        log_every=log_every,
    )


# ── CLI wrapper ─────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    compensate_lora_from_paths(
        teacher_path=args.teacher,
        student_path=args.student,
        student_quant=args.student_quant,
        teacher_quant=args.teacher_quant,
        calibration_data=args.calibration_data,
        output=args.output,
        steps=args.steps,
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        target_modules=args.target_modules,
        max_length=args.max_length,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
