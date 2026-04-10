# sentinel-ai forge worker
# GPU-accelerated model forging: prune, train, eval, deliver
#
# Build:  docker build -t continuum-forge .
# Run:    docker run --gpus all -v ./output:/app/output -v ~/.cache/huggingface:/root/.cache/huggingface continuum-forge <alloy.json>
#
# CPU-only (Mac/no GPU):
#   docker build --build-arg BASE_IMAGE=ubuntu:24.04 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t continuum-forge .

ARG BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

# Create venv (no more --break-system-packages)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch — index URL switches between CUDA and CPU builds
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir \
    torch --index-url ${TORCH_INDEX}

# ML dependencies (pinned to what works on BigMama)
RUN pip install --no-cache-dir \
    transformers==5.3.0 \
    datasets==4.8.4 \
    accelerate==1.13.0 \
    peft==0.18.1 \
    bitsandbytes==0.49.2 \
    safetensors==0.7.0 \
    huggingface_hub==1.8.0 \
    evalplus==0.3.1 \
    numpy tqdm matplotlib qrcode[pil]

WORKDIR /app

# Copy sentinel-ai source (see .dockerignore for exclusions)
COPY scripts/ scripts/
COPY *.py ./

# Models and output mount points
VOLUME ["/app/output", "/root/.cache/huggingface"]

ENTRYPOINT ["python3", "scripts/alloy_executor.py"]
