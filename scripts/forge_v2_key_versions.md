# forge v2 working environment — load-bearing package versions

Captured from BigMama after the v1.5 pad-upscale + Layer 7 gate + vLLM install
landed cleanly on 2026-04-07. Use this when restoring or moving the environment
to a different host.

## Critical pins

| Package | Version | Why it matters |
|---|---|---|
| vllm | 0.19.0 | Fast EvalPlus + 5090 (Blackwell) compute capability 12.0 supported |
| torch | 2.10.0 | Required by vllm 0.19; pulls in CUDA 12.8 wheels |
| torchvision | 0.25.0 | matches torch 2.10 |
| torchaudio | 2.10.0 | matches torch 2.10 |
| transformers | 4.57.6 | downgraded from 5.3.0 by vllm install; v1.5 pad-upscale verified to work on this version |
| accelerate | 1.13.0 | required by transformers 4.57; note that the evalplus hf-backend on 14B fp16 hits an offload-vs-.to() bug here |
| bitsandbytes | 0.49.2 | needed by forge_model.py auto-quant tier C path |
| peft | 0.18.1 | LoRA training in forge_model.py |
| safetensors | 0.7.0 | model save/load |
| gguf | 0.18.0 | scripts/stream_dequant.py and scripts/v1_to_v15_pad_upscale.py — read GGUF directly |
| evalplus | 0.3.1 | HumanEval+ + MBPP+ benchmarks; vLLM backend used (hf backend broken on 14B due to accelerate bug) |
| numpy | 2.2.6 | upgraded by vllm install; older code that uses np.float_ etc. needs review |
| datasets | 4.8.4 | training data loader |

## Known issues with this combination

1. **evalplus hf-backend crashes on 14B fp16** with `RuntimeError: You can't move a model that has some modules offloaded to cpu or disk.` Root cause: accelerate's `from_pretrained(device_map="auto")` triggers offload, then `HuggingFaceDecoder.__init__` calls `.to(device)` which is illegal after offload. Workaround: use vllm backend instead. Fix would be in `evalplus/provider/hf.py:45`.

2. **System Python is PEP 668 externally-managed** on this BigMama (Ubuntu/Debian). All `pip install` commands need `--user --break-system-packages`. The user-site dir is `/home/joel/.local/lib/python3.12/site-packages/`.

3. **transformers downgrade from 5.3.0 → 4.57.6** dragged in by vLLM. The forge_model.py and defrag_inline.py code paths have been verified to still work. If anything regresses, check `from_pretrained` calls — the 4.x → 5.x transformers had API changes in `attn_implementation`, `dtype`, etc.

## Restore command

```bash
# On a fresh BigMama-like Ubuntu host with CUDA 12.8 already installed:
pip install --user --break-system-packages -r forge_v2_requirements.txt
```

If pip refuses, the venv path is:

```bash
python3 -m venv ~/forge_v2_venv
~/forge_v2_venv/bin/pip install -r forge_v2_requirements.txt
source ~/forge_v2_venv/bin/activate
```

## Hardware constraints

- BigMama: WSL2 on Windows, single RTX 5090 (32 GB VRAM, compute capability 12.0), 31 GB system RAM, ~214 GB free disk
- The 31 GB system RAM is the bottleneck for `transformers.from_pretrained(..., gguf_file=...)` on the 14B model — that loader stages dequantized tensors on CPU before moving to GPU, OOMs around tensor 357/579. Use `scripts/stream_dequant.py` instead, which streams tensor-by-tensor on the GPU.
