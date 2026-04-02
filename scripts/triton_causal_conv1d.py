"""
Triton causal conv1d — drop-in for causal-conv1d CUDA package.
Works on RTX 5090 (sm_120) because Triton compiles for any GPU.
"""
import torch
import triton
import triton.language as tl
import time
import torch.nn.functional as F


@triton.jit
def _conv1d_fwd(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    seqlen, width: tl.constexpr, dim,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < seqlen
    base = pid_b * dim * seqlen + pid_d * seqlen
    bw = pid_d * width
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k in tl.static_range(4):
        if k < width:
            wv = tl.load(w_ptr + bw + k)
            ip = offs - (width - 1 - k)
            im = (ip >= 0) & (ip < seqlen) & mask
            xv = tl.load(x_ptr + base + ip, mask=im, other=0.0)
            acc += xv * wv
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + pid_d)
    tl.store(out_ptr + base + offs, acc, mask=mask)


def triton_causal_conv1d(x, weight, bias=None, activation=None, **kwargs):
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    out = torch.empty_like(x)
    BN = min(512, triton.next_power_of_2(seqlen))
    grid = (batch, dim, triton.cdiv(seqlen, BN))
    _conv1d_fwd[grid](x, weight, bias, out, seqlen=seqlen, width=width, dim=dim, BLOCK_N=BN)
    if activation == "silu":
        out = torch.nn.functional.silu(out)
    return out


def triton_causal_conv1d_update(x, conv_state, weight, bias=None, activation=None):
    """Stateful update for autoregressive generation (single token step).
    x: (batch, dim, 1) or (batch, dim)
    conv_state: (batch, dim, width) — sliding window state
    weight: (dim, width)
    Returns: (output, new_conv_state)
    """
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    batch, dim, _ = x.shape
    width = weight.shape[1]

    # Shift state left and append new input
    new_state = torch.roll(conv_state, -1, dims=-1)
    new_state[:, :, -1] = x[:, :, 0]

    # Depthwise conv: sum(state * weight) per channel
    out = (new_state * weight.unsqueeze(0)).sum(dim=-1)  # (batch, dim)
    if bias is not None:
        out = out + bias

    if activation == "silu":
        out = torch.nn.functional.silu(out)

    return out.unsqueeze(-1), new_state


def monkey_patch():
    import types, sys, importlib
    # Create proper module with __spec__ so importlib.util.find_spec works
    m = types.ModuleType("causal_conv1d")
    m.__spec__ = importlib.machinery.ModuleSpec("causal_conv1d", None)
    m.__path__ = []
    m.__package__ = "causal_conv1d"
    m.causal_conv1d_fn = triton_causal_conv1d
    m.causal_conv1d_update = triton_causal_conv1d_update
    sys.modules["causal_conv1d"] = m

    mi = types.ModuleType("causal_conv1d.causal_conv1d_interface")
    mi.__spec__ = importlib.machinery.ModuleSpec("causal_conv1d.causal_conv1d_interface", None)
    mi.__package__ = "causal_conv1d"
    mi.causal_conv1d_fn = triton_causal_conv1d
    mi.causal_conv1d_update = triton_causal_conv1d_update
    sys.modules["causal_conv1d.causal_conv1d_interface"] = mi
    print("[triton_causal_conv1d] Patched — Triton kernel active")


if __name__ == "__main__":
    batch, dim, seqlen, width = 1, 2560, 1024, 4
    x = torch.randn(batch, dim, seqlen, device="cuda", dtype=torch.float16)
    w = torch.randn(dim, width, device="cuda", dtype=torch.float16)
    b = torch.randn(dim, device="cuda", dtype=torch.float16)

    out_triton = triton_causal_conv1d(x, w, b)
    out_torch = F.conv1d(x, w.unsqueeze(1), b, padding=width - 1, groups=dim)[..., :seqlen]
    diff = (out_triton - out_torch).abs().max().item()
    ok = "PASS" if diff < 0.01 else "FAIL"
    print(f"Correctness: {ok} (max diff {diff:.6f})")

    for _ in range(5):
        triton_causal_conv1d(x, w, b)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        triton_causal_conv1d(x, w, b)
    torch.cuda.synchronize()
    tt = (time.time() - t0) / 100

    for _ in range(5):
        F.conv1d(x, w.unsqueeze(1), b, padding=width - 1, groups=dim)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        F.conv1d(x, w.unsqueeze(1), b, padding=width - 1, groups=dim)[..., :seqlen]
    torch.cuda.synchronize()
    tf = (time.time() - t0) / 100

    print(f"Triton: {tt*1000:.2f}ms  Torch: {tf*1000:.2f}ms  Speedup: {tf/tt:.1f}x")
