#!/usr/bin/env bash
# Continuum Forge — one-command install
# curl -fsSL https://cambriantech.github.io/continuum/install.sh | bash
set -euo pipefail

REGISTRY="ghcr.io/cambriantech"
FORGE_IMAGE="$REGISTRY/forge-worker:latest"

info()  { echo -e "\033[1;36m→\033[0m $*"; }
ok()    { echo -e "\033[1;32m✓\033[0m $*"; }
warn()  { echo -e "\033[1;33m!\033[0m $*"; }
fail()  { echo -e "\033[1;31m✗\033[0m $*"; exit 1; }

# ── Detect environment ──────────────────────────────────────
info "Detecting environment..."

OS="$(uname -s)"
ARCH="$(uname -m)"
HAS_GPU=false

case "$OS" in
  Linux)
    if command -v nvidia-smi &>/dev/null; then
      HAS_GPU=true
      GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
      ok "GPU detected: $GPU_NAME"
    else
      warn "No NVIDIA GPU detected — forge will run on CPU (slow)"
    fi
    ;;
  Darwin)
    warn "macOS — no NVIDIA GPU support, forge jobs should run on a GPU machine"
    ;;
  *)
    fail "Unsupported OS: $OS"
    ;;
esac

ok "Platform: $OS $ARCH"

# ── Install Docker if missing ───────────────────────────────
if ! command -v docker &>/dev/null; then
  info "Docker not found — installing..."
  case "$OS" in
    Linux)
      curl -fsSL https://get.docker.com | sh
      sudo usermod -aG docker "$USER"
      warn "Added $USER to docker group — you may need to log out and back in"
      ;;
    Darwin)
      fail "Install Docker Desktop from https://docker.com/products/docker-desktop and re-run this script"
      ;;
  esac
fi

ok "Docker $(docker --version | grep -oP '\d+\.\d+\.\d+')"

# ── Install NVIDIA Container Toolkit (Linux + GPU only) ─────
if [[ "$HAS_GPU" == "true" ]]; then
  if ! docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi &>/dev/null 2>&1; then
    info "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ok "NVIDIA Container Toolkit installed"
  else
    ok "NVIDIA Container Toolkit already configured"
  fi
fi

# ── Pull forge image ────────────────────────────────────────
info "Pulling forge worker image (this may take a few minutes on first run)..."
docker pull "$FORGE_IMAGE"
ok "Forge image ready"

# ── Set up workspace ────────────────────────────────────────
FORGE_DIR="${FORGE_DIR:-$HOME/continuum-forge}"
mkdir -p "$FORGE_DIR/output" "$FORGE_DIR/alloys"

# Write convenience script
cat > "$FORGE_DIR/forge" << 'FORGE_SCRIPT'
#!/usr/bin/env bash
# Usage: ./forge <alloy.json> [--output-dir <dir>] [--dry-run]
set -euo pipefail

FORGE_IMAGE="ghcr.io/cambriantech/forge-worker:latest"
FORGE_DIR="$(cd "$(dirname "$0")" && pwd)"

GPU_FLAGS=""
if command -v nvidia-smi &>/dev/null; then
  GPU_FLAGS="--gpus all"
fi

docker run --rm -it $GPU_FLAGS \
  -v "$FORGE_DIR/output:/app/output" \
  -v "$FORGE_DIR/alloys:/app/alloys:ro" \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
  "$FORGE_IMAGE" "$@"
FORGE_SCRIPT
chmod +x "$FORGE_DIR/forge"

ok "Workspace: $FORGE_DIR"

# ── Done ────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Continuum Forge installed successfully"
echo ""
echo "  Forge a model:"
echo "    cd $FORGE_DIR"
echo "    cp your-recipe.alloy.json alloys/"
echo "    ./forge alloys/your-recipe.alloy.json --output-dir output/my-model"
echo ""
if [[ "$HAS_GPU" == "true" ]]; then
  echo "  GPU: $GPU_NAME"
else
  echo "  No GPU — jobs will run on CPU (use a GPU machine for real forging)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
