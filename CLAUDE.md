# CLAUDE.md - AI Agent Guidelines

## Build Commands
- Train model: `python train.py --model_name MODEL --dataset DATASET --epochs N --batch_size B --lr LR --device {cpu,cuda}`
- Run inference: `python main.py --model_name MODEL --prompt "text" --max_length L --device {cpu,cuda} --temperature T`
- Validate notebook: `python notebooks/validate_notebook.py notebook_path`
- Fix notebook metadata: `python notebooks/fix_notebook_metadata.py`
- Test pruning: `python scripts/inference_with_pruning.py --strategy entropy --pruning_level 0.5 --prompt "Your text"`
- Test model support: `python test_model_support.py --device {cpu,cuda} --verbose`

## Multi-Model Support
- Test GPT-2: `python main.py --model_name distilgpt2 --prompt "Your prompt here"`
- Test OPT: `python main.py --model_name facebook/opt-125m --prompt "Your prompt here"`
- Test Pythia: `python main.py --model_name EleutherAI/pythia-70m --prompt "Your prompt here"`
- Test BLOOM: `python main.py --model_name bigscience/bloom-560m --prompt "Your prompt here"`
- Test Llama: `python main.py --model_name meta-llama/Llama-2-7b-hf --prompt "Your prompt here"`
- Generate samples: `python generate_samples.py`
- Fine-tune pruned model: `python scripts/finetune_pruned_model.py --model_path PATH --dataset DATASET --output_path OUTPUT --enable_head_lr`
- Test multi-model support: `python scripts/test_multi_model_support.py --models gpt2,opt,pythia,bloom`
- Multi-model profiling: `./scripts/run_multi_model_profile.sh --models "gpt2,distilgpt2,bigscience/bloom-560m" --device cpu`

## Cache Management
- Clean all Hugging Face caches: `rm -rf ~/.cache/huggingface`
- View cache contents: `huggingface-cli scan-cache`
- Clean specific model: `rm -rf ~/.cache/huggingface/hub/models--MODEL_NAME`

## Code Style Guidelines
- Imports: standard library → third-party → local modules
- Naming: snake_case for functions/variables, CamelCase for classes, UPPER_CASE for constants
- Error handling: Use try/except with specific exceptions and meaningful error messages
- Parameters: Provide sensible defaults where appropriate
- Documentation: Add comments for complex logic, use argparse with helpful descriptions
- Organization: Keep functions single-purpose and modular

## Project Structure
- Models in `models/`, with model loaders for different architectures in `models/loaders/`
- Controller logic in `controller/`, utilities in `utils/`
- Main training logic in `train.py`, inference in `main.py`
- Notebooks in `notebooks/` demonstrate various model behaviors and visualizations
- Scripts in `scripts/` provide benchmarking, fine-tuning, and specialized tools
- Documentation in `docs/` includes methodology, fine-tuning guides, and technical details

## Branch Management
1. Always create feature or bug branches from main:
   - Feature branches: `git checkout -b feature/what-i-did`
   - Bug fix branches: `git checkout -b bugs/what-i-fixed`

2. Never commit directly to main except for small documentation changes.

3. Use meaningful branch names that describe what you're working on.

4. When creating a PR, include:
   - Clear summary of changes
   - Test plan with verification steps
   - Any notes on implementation details

5. After merging, always switch back to main before creating a new branch:
   ```bash
   git checkout main
   git pull
   git checkout -b feature/new-feature
   ```

## Commit Guidelines
1. Keep commits focused on single logical changes.
2. Use descriptive commit messages with a clear title line.
3. Include "🤖 Generated with [Claude Code]" tag in commits.
4. Include co-author attribution.

## Workflow Commands
```bash
# Create a new feature branch
git checkout main
git pull
git checkout -b feature/my-feature

# Stage and commit changes
git add <files>
git commit -m "Add my feature

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push and create PR
git push -u origin feature/my-feature
gh pr create --title "Add my feature" --body "..."
```