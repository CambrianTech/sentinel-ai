#!/usr/bin/env bash
# sentinel-ai setup — works on CUDA (incl. 5090), MPS (Apple Silicon), and CPU
set -euo pipefail

cd "$(dirname "$0")"

echo "=== sentinel-ai setup ==="

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip -q

# Detect GPU and install appropriate PyTorch
install_pytorch() {
    # Check for NVIDIA GPU with CC 12.0+ (5090, etc.)
    if python3 -c "
import subprocess, re, sys
try:
    out = subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL).decode()
    # Also check via pytorch-independent CUDA query
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        cc = float(result.stdout.strip().split('\n')[0])
        if cc >= 12.0:
            print(f'CC {cc} — needs cu128+')
            sys.exit(0)
        elif cc >= 9.0:
            print(f'CC {cc} — cu126 OK')
            sys.exit(1)
        else:
            print(f'CC {cc} — cu124 OK')
            sys.exit(2)
except FileNotFoundError:
    sys.exit(10)  # No nvidia-smi
" 2>/dev/null; then
        echo "Installing PyTorch with CUDA 12.8 (CC 12.0+ GPU detected)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q
    elif [ $? -eq 1 ]; then
        echo "Installing PyTorch with CUDA 12.6..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 -q
    elif [ $? -eq 2 ]; then
        echo "Installing PyTorch with CUDA 12.4..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
    elif [ $? -eq 10 ]; then
        # No NVIDIA GPU — check for MPS (Apple Silicon) or fall back to CPU
        echo "No NVIDIA GPU detected. Installing CPU/MPS PyTorch..."
        pip install torch torchvision torchaudio -q
    fi
}

install_pytorch

# Install remaining dependencies
echo "Installing dependencies..."
pip install transformers datasets tqdm matplotlib accelerate seaborn -q

# Verify
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    x = torch.randn(100, 100, device='cuda')
    _ = x @ x.T
    print(f'GPU: {name} ({vram:.0f}GB) — CUDA verified')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    x = torch.randn(100, 100, device='mps')
    _ = x @ x.T
    print(f'Apple Silicon MPS — verified')
else:
    print('CPU only')
print('Ready.')
"
