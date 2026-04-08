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

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model

# Same-directory import; the harness_checks module enforces the no-fallback
# discipline at the precondition layer.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_checks import assert_explicit_head_dim  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _bnb_config(quant: str) -> BitsAndBytesConfig | None:
    """Build a BitsAndBytesConfig for the named quant tier, or None for fp16."""
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


def load_teacher(path: str, device: str, quant: str = "8bit") -> torch.nn.Module:
    """Load the unpruned teacher, frozen.

    quant ∈ {"8bit","4bit"}. Teacher is forward-pass-only (no gradients) so
    quantization is essentially free for distillation logits. For 7B-class
    teachers 8bit fits comfortably alongside an fp16 student on a 32 GB GPU.
    For 30B-class teachers 4bit is required because the student also has to
    fit in 4bit + LoRA + activations + KV cache simultaneously.
    """
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


def load_student(path: str, device: str, quant: str = "fp16") -> torch.nn.Module:
    """Load the pruned student, with gradient checkpointing enabled.

    quant ∈ {"fp16","4bit"}. fp16 is the default for ≤14B-class students;
    backprop runs through the full-precision weights and the LoRA is fp16.

    For 30B-class students, fp16 (~40 GB) doesn't fit a 32 GB GPU at all,
    so 4bit (NF4 + double quant) is required. In 4bit mode the student is
    QLoRA-style: base weights are frozen 4bit, LoRA adapters are fp16, and
    only the LoRA parameters are trained. `prepare_model_for_kbit_training`
    handles the dtype dance and the gradient checkpointing setup.
    """
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
    student: torch.nn.Module,
    rank: int,
    alpha: int,
    target_modules: list[str],
    dropout: float = 0.05,
) -> torch.nn.Module:
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
    teacher_hidden: tuple[torch.Tensor, ...],
    student_hidden: tuple[torch.Tensor, ...],
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    loss_type: str,
    temperature: float = 2.0,
) -> torch.Tensor:
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


class JsonlTextDataset(Dataset):
    """Calibration dataset from a JSONL file with ``{"text": "..."}`` entries.

    The caller is responsible for constructing the JSONL with held-out task
    examples (HumanEval problems, GSM8K, MMLU, or a mixture). This is the
    held-out-aware calibration data per §4.1.3.2's structural fix; it must
    not be drawn from the student's fine-tuning corpus or the calibration-
    distribution narrowness will recur.
    """

    def __init__(self, path: str, tokenizer, max_length: int = 1024):
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


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[compensation_lora] device={device}")
    print(f"[compensation_lora] loading teacher from {args.teacher} (bnb-{args.teacher_quant}, frozen)")
    teacher = load_teacher(args.teacher, device, quant=args.teacher_quant)

    print(f"[compensation_lora] loading student from {args.student} ({args.student_quant}, grad-checkpointed)")
    student = load_student(args.student, device, quant=args.student_quant)

    # Preconditions: the STUDENT must carry head_dim explicitly (post-defrag
    # invariant from §4.1.3 / Finding 6 — the v1 bug we are explicitly
    # defending against). The TEACHER does NOT need this assertion: an
    # unmodified base model legitimately has implicit head_dim because no
    # defrag happened, and `hidden_size / num_attention_heads` is the
    # ground-truth for the unmodified architecture. The assertion exists to
    # catch the case where a forge artifact loses head_dim during save_pretrained,
    # which only applies to artifacts that went through defrag.
    print("[compensation_lora] checking preconditions")
    assert_explicit_head_dim(student.config)
    # Teacher + student must have matching hidden_size + num_hidden_layers
    # for hidden-state distillation to be coherent (per-layer alignment).
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
        f"(r={args.lora_r}, alpha={args.lora_alpha}, "
        f"target_modules={args.target_modules})"
    )
    student = attach_compensation_lora(
        student=student,
        rank=args.lora_r,
        alpha=args.lora_alpha,
        target_modules=args.target_modules,
    )
    student.print_trainable_parameters()

    print(f"[compensation_lora] loading tokenizer from {args.teacher}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[compensation_lora] loading calibration data from {args.calibration_data}")
    dataset = JsonlTextDataset(
        path=args.calibration_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    print(f"[compensation_lora] loaded {len(dataset)} calibration examples")

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    print(f"[compensation_lora] training for {args.steps} steps with loss={args.loss_type}")
    student.train()
    step = 0
    epoch = 0
    while step < args.steps:
        epoch += 1
        for example in dataset:
            if step >= args.steps:
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
                loss_type=args.loss_type,
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(
                    f"[compensation_lora] epoch {epoch} step {step}/{args.steps} "
                    f"loss={loss.item():.6f}"
                )

    print("[compensation_lora] training complete; merging LoRA into student weights")
    student = student.merge_and_unload()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compensation_lora] saving compensated student to {out_dir}")
    student.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    # Verify the saved config still carries head_dim explicitly. This catches
    # the same class of bug as v1's tokenizer_config.json drop in §4.1.1
    # Failure 4: post-save metadata loss that's invisible until reload.
    saved_config = AutoConfig.from_pretrained(out_dir)
    assert_explicit_head_dim(saved_config)

    print(f"[compensation_lora] done. compensated model at {out_dir}")
    print(
        "[compensation_lora] next step: run the calibrated EvalPlus pipeline against "
        "this artifact and compare HumanEval pass@1 to the un-compensated student"
    )


if __name__ == "__main__":
    main()
