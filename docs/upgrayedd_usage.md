# upgrayedd.py - Give Your Models an Upgrade

>  💡 "Spelled with two D's for a double dose of adaptive optimization."

`upgrayedd.py` is a user-friendly command-line tool that transforms any HuggingFace model into an adaptive, self-optimizing neural network using Sentinel-AI's neural plasticity and controller systems.

## 🔥 Features

- **One Command Transformation**: Update any HuggingFace model with a single command
- **Neural Plasticity Cycles**: Automatic pruning, measuring, growing, and learning loops
- **Controller-Guided Optimization**: ANN controller that dynamically adjusts the model architecture
- **Self-Optimization**: Models continuously improve through feedback loops
- **Efficient Fine-Tuning**: Uses differential learning rates for faster adaptation
- **HuggingFace Compatible**: Upgraded models remain compatible with the HuggingFace ecosystem

## 📋 Basic Usage

```bash
# Basic usage with GPT-2
python upgrayedd.py --model distilgpt2 --dataset tiny_shakespeare

# Adjust pruning and cycles
python upgrayedd.py --model gpt2 --pruning-level 0.3 --cycles 10

# Try with OPT models
python upgrayedd.py --model facebook/opt-125m --growth-ratio 0.6

# Use with Llama-2
python upgrayedd.py --model meta-llama/Llama-2-7b-hf --controller ann
```

## 🛠️ Command Line Options

### Model and Dataset Parameters

- `--model` - HuggingFace model name or path (e.g., distilgpt2, gpt2, facebook/opt-125m)
- `--dataset` - Dataset to use for optimization (default: tiny_shakespeare)
- `--output-dir` - Directory to save the upgraded model and results (default: ./output/upgrayedd)

### Optimization Parameters

- `--cycles` - Number of plasticity cycles to run (default: 5)
- `--pruning-level` - Initial pruning level (default: 0.3 = 30% of heads)
- `--growth-ratio` - Growth ratio for pruned heads (default: 0.5 = 50% of pruned heads)
- `--learning-rate` - Learning rate for fine-tuning (default: 5e-5)
- `--controller` - Controller type [ann, rule] (default: ann)

### Inference Parameters

- `--run-inference` - Run inference with the upgraded model after optimization
- `--prompt` - Prompt for inference (default: None)
- `--temperature` - Temperature for inference (default: 0.7)

### System Parameters

- `--device` - Device to use [cuda, cpu] (default: cuda if available, else cpu)
- `--save-checkpoints` - Save model checkpoints after each optimization cycle
- `--skip-optimization` - Skip the optimization phase (just inject adaptive modules)
- `--verbose` - Enable verbose output

## 🧩 How It Works

The upgrading process consists of the following steps:

1. **Loading**: Load the base model and tokenizer from HuggingFace
2. **Injection**: Transform the standard model into an adaptive one by:
   - Adding attention gates
   - Setting up the neural controller
   - Preparing pruning and growth components
3. **Optimization**: Run multiple cycles of:
   - Prune less important attention heads
   - Measure the impact on model performance
   - Grow back critical heads strategically
   - Fine-tune the model with differential learning rates
4. **Feedback**: The system continuously improves by:
   - Learning from successful transformations
   - Adjusting strategies based on results
   - Detecting and correcting output degeneration
5. **Saving**: Create a compatible, optimized model that:
   - Has better performance
   - Is more efficient (fewer parameters)
   - Maintains HuggingFace compatibility

## 📊 Output

The script produces the following outputs:

- **Upgraded Model**: The optimized model saved in the output directory
- **Metrics**: Detailed performance metrics at each optimization stage
- **Logs**: Comprehensive logs of the entire process
- **Visualizations**: (If supported in your installation) Visualizations of how the model changed

## 🔧 Advanced Usage

### Running with Custom Datasets

```bash
# Use a custom dataset
python upgrayedd.py --model distilgpt2 --dataset my_custom_dataset

# Skip dataset for inference-only mode
python upgrayedd.py --model gpt2 --skip-optimization --run-inference
```

### Controlling the Optimization Process

```bash
# More aggressive pruning
python upgrayedd.py --model gpt2 --pruning-level 0.5

# More conservative growth
python upgrayedd.py --model gpt2 --growth-ratio 0.2

# More training cycles
python upgrayedd.py --model gpt2 --cycles 15
```

### Inference with Upgraded Models

```bash
# Run inference after optimization
python upgrayedd.py --model gpt2 --run-inference --prompt "The future of AI is"

# Just run inference with a previously upgraded model
python upgrayedd.py --model ./output/upgrayedd/gpt2_20230101-120000/hf_model --skip-optimization --run-inference
```

## 🔍 Examples

### Basic Optimization

```bash
python upgrayedd.py --model distilgpt2 --dataset tiny_shakespeare
```

Output:
```
=== Starting upgrade process for distilgpt2 ===
 _   _                                     _     _ 
| | | |                                   | |   | |
| | | |_ __   __ _ _ __ __ _ _   _  ___  _| | __| |
| | | | '_ \ / _` | '__/ _` | | | |/ _ \/ _ |/ _` |
| |_| | |_) | (_| | | | (_| | |_| |  __/ (_| | (_| |
 \___/| .__/ \__, |_|  \__,_|\__, |\___|\__,_|\__,_|
      | |     __/ |           __/ |                 
      |_|    |___/           |___/                  

  Spelled with two D's for a double dose of adaptive optimization

Loading model and tokenizer for distilgpt2...
Successfully loaded distilgpt2
Injecting adaptive modules into model...
Successfully injected adaptive modules
Loading dataset: tiny_shakespeare...
Successfully loaded dataset: tiny_shakespeare
Setting up neural plasticity optimization cycle...
Running optimization for 5 cycles...

... [optimization details] ...

==================================================
🎉 Model successfully upgraded!
Output directory: ./output/upgrayedd/distilgpt2_20230101-120000
==================================================
```

### Comparing Before and After

```bash
python upgrayedd.py --model gpt2 --run-inference --prompt "The key to artificial intelligence is"
```

Output:
```
... [optimization details] ...

Running inference with prompt: The key to artificial intelligence is
Generated text: The key to artificial intelligence is combining neural plasticity 
with controlled adaptation. By allowing networks to prune unnecessary connections 
while strategically growing critical pathways, AI systems can continuously evolve 
to handle new tasks while maintaining efficiency and performance.

==================================================
🎉 Model successfully upgraded!
Output directory: ./output/upgrayedd/gpt2_20230101-120000
==================================================
```

## 🤝 Contributing

Interested in contributing to Sentinel-AI and its tools? Check out:

1. Our [GitHub repository](https://github.com/your-org/sentinel-ai)
2. The [contribution guidelines](https://github.com/your-org/sentinel-ai/blob/main/CONTRIBUTING.md)
3. The [list of open issues](https://github.com/your-org/sentinel-ai/issues)

## 📃 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*With upgrayedd.py, your models aren't just trained, they're "upgrayedd-ed" with a double dose of adaptive optimization.*