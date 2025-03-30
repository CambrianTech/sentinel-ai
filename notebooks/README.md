
---

# 🧪 Adaptive Transformer Notebooks

This directory contains interactive Jupyter notebooks designed for experimenting with, visualizing, and validating the capabilities of the **Adaptive Transformer** model. The notebooks demonstrate the unique features of the model, including dynamic head pruning and growth, controller-based adaptivity, and U-Net-inspired adaptivity.


## 📚 Available Notebooks

| Notebook | Description |
| -------- | ----------- |
| [`AdaptiveTransformerNotebook.ipynb`](./AdaptiveTransformerNotebook.ipynb) | Full Colab-ready notebook for training, inference, and benchmarking the Adaptive Transformer. Includes visualizations of metrics like perplexity, active head counts, and gate activations over time. |
| [`AdaptiveTransformer_Proof_of_Adaptivity.ipynb`](./AdaptiveTransformer_Proof_of_Adaptivity.ipynb) | Demonstrates mathematically and empirically that the adaptive mechanism (head pruning and addition) works as intended. |
| [`AdaptiveTransformer_UNet_Adaptivity.ipynb`](./AdaptiveTransformer_UNet_Adaptivity.ipynb) | Showcases dynamic adaptive behavior using a U-Net-inspired mechanism for growing and pruning attention heads during training. |
| [`HeadPruningEffectiveness.ipynb`](./HeadPruningEffectiveness.ipynb) | Evaluates the effectiveness of head pruning on model performance, comparing metrics before and after pruning. |
| [`AdaptiveBehaviorVisualization.ipynb`](./AdaptiveBehaviorVisualization.ipynb) | Interactive visualizations of how the adaptive gating mechanism evolves during training, illustrating gate activation dynamics and head utilization. |
| [`AdaptiveAttentionVisualization.ipynb`](./AdaptiveAttentionVisualization.ipynb) | Side-by-side visualizations of attention patterns across layers and heads for baseline vs. adaptive models. |
| [`AdaptiveAttentionHeatmapComparison.ipynb`](./AdaptiveAttentionHeatmapComparison.ipynb) | Visual comparison of attention heatmaps between adaptive and non-adaptive models, demonstrating how adaptive gating influences attention distribution. |
| [`CheckpointResumptionTest.ipynb`](./CheckpointResumptionTest.ipynb) | Verifies that training resumes correctly from checkpoints and that gate values continue evolving. |
| [`LowResourceAdaptivity.ipynb`](./LowResourceAdaptivity.ipynb) | Proves that the model remains functional under low compute settings (e.g., Colab T4) and adapts to limited resources by pruning itself effectively. |
| [`ControllerDynamics.ipynb`](./ControllerDynamics.ipynb) | Tracks the evolution of the controller’s internal logits and their impact on head usage over time. |
| [`ModelScalingTest.ipynb`](./ModelScalingTest.ipynb) | Compares adaptive training performance across model sizes (e.g., `distilgpt2`, `gpt2`, `gpt2-medium`). |

---

## 🔧 Running Notebooks Locally

Ensure your current directory is the repository root:

```bash
cd sentinel-ai
jupyter notebook notebooks/
```

Make sure dependencies are installed:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running Notebooks on Google Colab

1. Open a new Google Colab session and clone this repository:

```python
!git clone https://github.com/your-username/sentinel-ai.git
import sys
sys.path.append('/content/sentinel-ai')

# Navigate to the notebooks directory
%cd /content/sentinel-ai/notebooks
```

2. Install requirements:

```python
!pip install -r ../requirements.txt
```

3. Open and run any notebook directly within Colab.

---

## 💡 Tips

- All notebooks auto-download required datasets and models via Hugging Face APIs.
- Checkpoints are saved regularly, allowing training sessions to resume seamlessly.
- If running on Colab with GPU, mixed-precision training is automatically leveraged for faster training.

---

> 📁 **Prefer scripts?**  
> Check [`/scripts/train_colab.py`](../scripts/train_colab.py) and [`/scripts/eval_colab.py`](../scripts/eval_colab.py) for non-notebook alternatives providing similar functionality.

---