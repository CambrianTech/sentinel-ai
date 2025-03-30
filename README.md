# 👾 Sentinel-AI — Adaptive Transformer with ANN Controller

Welcome to **Sentinel-AI**, a research framework for adaptive transformer models that restructure themselves in real-time. This architecture introduces:

- **Sentinel Gating**: Dynamic head pruning and regrowth  
- **ANN Controller**: Guides adaptivity using gradient and entropy signals  
- **Plug-and-Play Flexibility**: Load and upgrade pretrained models like `GPT2`, `DistilGPT2`, and others

<p align="center">
  <img src="./docs/assets/architecture_full_diagram.png" width="1000"/>
</p>

👾 How U-Net Works in Our Transformer
In Sentinel-AI, we borrow from U-Net’s idea of hierarchical skip pathways to stabilize regrowth of pruned attention heads and preserve earlier semantic features. Here's how:

🔄 U-Net Adaptivity in Transformer:
Skip Paths: Low-level gate activations or embeddings from earlier layers are forwarded to later layers.

Controller Memory: The ANN controller can receive both local and skip-connected signals (e.g., early entropy values).

Reinforcement Signal: When heads regrow, they can be initialized or influenced by past behavior — like how U-Net reuses encoder features for decoder guidance.

This is especially valuable when a head is pruned and later re-activated — we want the new head to resume useful behavior, not start from scratch.


🔬 Our work builds on the premise that models should be able to **grow**, **shrink**, and **adapt** to task complexity on demand — ideal for edge deployment, low-resource training, and continual learning.

📄 **[Read our Paper](./paper/adaptive_transformer_with_controller.md)**  
🧪 **[Explore the Interactive Notebooks](./notebooks/)**

---

## 🚀 Key Features

- 🔁 **Dynamic Adaptivity** — Models automatically prune or re-enable attention heads during training  
- 🎛️ **Controller-Driven Learning** — An ANN learns to adjust gates using runtime statistics  
- 🪜 **U-Net Style Growth** — Temporal skip connections stabilize regrowth of attention units  
- ⚡ **Colab & Low-resource Ready** — Optimized for T4-class GPUs and RAM-constrained settings  
- 🧩 **Compatible with Hugging Face** — Import pretrained `GPT2`, `DistilGPT2`, and others for rapid experimentation  

---


## 🗂️ Repository Structure

```bash
sentinel-ai/
├── models/                # Core model + adapters
├── controller/            # ANN Controller for head gating
├── datasets/              # Tokenization, batching, evaluation
├── utils/                 # Logging, training logic, wrappers
├── notebooks/             # Exploratory analysis and visualization
├── paper/                 # Research paper in Markdown
├── scripts/               # Colab-optimized training/eval
├── train.py               # CLI for training
├── main.py                # CLI for inference
└── requirements.txt       # Environment dependencies
```

---

## 🧪 Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

This launches adaptive training using GPT-style weights and your selected dataset. Head activations are tracked and updated dynamically using controller signals.

### Inference

```bash
python main.py
# or with a different model
MODEL_NAME=gpt2 python main.py
```

### Use on Google Colab

```python
# In Colab
!git clone https://github.com/your-username/sentinel-ai.git
%cd sentinel-ai
!pip install -r requirements.txt
```

You can then open any notebook in `/notebooks/` or run `train_colab.py`.

---

## 📊 Visual Analysis & Evaluation Notebooks

We’ve included a rich suite of notebooks that explore different aspects of the architecture:

| Notebook | Description |
|----------|-------------|
| **AdaptiveTransformerNotebook** | Full Colab-ready training + benchmarking notebook |
| **Proof of Adaptivity** | Demonstrates head growth and pruning over time |
| **UNet Adaptivity** | Shows skip-residual inspired head adaptation |
| **Controller Dynamics** | Tracks controller gate logits and updates |
| **Attention Heatmaps** | Compare attention layers side-by-side |
| **Checkpoint Resumption** | Verifies full recovery from saved state |
| **Low Resource Adaptivity** | Validates self-slimming behavior under compute constraints |
| **Model Scaling Test** | Evaluate across GPT2 variants (`distilgpt2`, `gpt2`, etc.) |

📁 [See full list in `notebooks/`](./notebooks/README.md)

---

## 🧠 How It Works (Architecture)

```
              ┌─────────────────────────────────────┐
              │  Pretrained GPT-like Transformer    │
              └─────────────────────────────────────┘
                             │
                             ▼
                ┌──────────────────────┐
                │ Sentinel Gates (Per-Head) ──┐
                └──────────────────────┘     │
                          ▲                  ▼
               ┌──────────────────────┐  ┌────────────┐
               │    ANN Controller    │  │ Attention  │
               └──────────────────────┘  └────────────┘
                          │
                          ▼
              (Dynamic activation / pruning)
```

---

## 🧠 Datasets Supported

- 📝 **Tiny Shakespeare**
- 📚 **WikiText-2**
- 🌐 **OpenWebText**

(Selected in `dataset_loader.py` or via notebook configuration.)

---

## ✅ Checkpointing

```python
from utils.checkpoint import save_checkpoint, load_checkpoint

# Save
save_checkpoint("checkpoint.pth", model, optimizer, head_lr_multipliers, epoch, step)

# Load
load_checkpoint("checkpoint.pth", model, optimizer)
```

---

## 📌 Future Directions

- 🤖 Expand controller to use gradient-based attention attribution
- 🧬 Enable continual learning across tasks and domains
- 🔀 Experiment with LoRA and Adapter integration for rapid task adaptation
- 📡 Federated scaling of adaptive models across edge devices

---

## 👥 Contributing

We welcome contributions!  
- Found a bug? File an issue.  
- Want to extend the controller or improve training? Submit a PR.  
- Interested in building new visual notebooks? We'd love that too.

---

🧪 Built with care by researchers exploring model self-adaptation, efficient inference, and plasticity in deep networks.
