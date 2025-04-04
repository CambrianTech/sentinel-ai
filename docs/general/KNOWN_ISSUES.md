# Known Issues and Limitations

## Fine-Tuning Issues

### NaN Values During Training

Some models, particularly OPT models, may produce NaN values during training. We've implemented a NaN-safe loss function in the stability module, but it's still possible to encounter NaN values under certain conditions. If you experience NaN values during training:

1. Try using a smaller batch size
2. Reduce the learning rate further
3. Use a smaller model if possible

### Memory Limitations in Colab

Despite our memory optimization efforts, some large models (particularly facebook/opt-1.3b and gpt2-xl) may still experience Out of Memory errors on T4 GPUs in Google Colab. We've implemented several strategies to manage memory, including:

- Dynamic batch size adjustment
- Automatic sequence length reduction
- Model-specific optimizations

However, some models are simply too large for the available GPU memory, especially when combined with pruning and fine-tuning. If you encounter memory errors:

1. Use smaller models when possible (facebook/opt-350m performs well)
2. Use higher pruning levels (0.3-0.5) to reduce the model size
3. Activate "High RAM" runtime in Colab settings

### DistilGPT2 Skipping

There's a known issue where the system incorrectly categorizes DistilGPT2 as exceeding memory limits. This is due to an error in our model filtering logic. We plan to fix this in a future update.

### Fine-Tuning Stability

We've improved the stability of fine-tuning through several mechanisms, but it's still possible to encounter instability, especially with larger models or aggressive pruning. Signs of instability include:

- Models generating repetitive special tokens (`<s><s><s>...`)
- NaN perplexity values
- Failure to generate coherent text after fine-tuning

We recommend trying different pruning strategies and lower learning rates if you encounter these issues.

## Colab Integration

### Environment Detection

The environment detection logic relies on available packages and may not always correctly identify the hardware configuration. In some cases, this can lead to suboptimal parameter settings. If you know your hardware configuration, you can manually set parameters like GPU memory in the code.

## Future Work

We're planning several improvements to address these issues:

1. Refactor the codebase into a more modular library structure
2. Improve test coverage with unit tests for all components
3. Add better logging and diagnostics
4. Implement more sophisticated memory management techniques
5. Provide more fine-grained control over optimization parameters

If you encounter any other issues, please report them in the repository issues section.