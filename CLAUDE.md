# CLAUDE.md - AI Agent Guidelines

## Claude Notes - Sentinel-AI Project
- ALWAYS increment version numbers when fixing bugs (e.g., v0.0.33 → v0.0.34)
- ALWAYS include full timestamps in version numbers with CURRENT YEAR 2025 (e.g., v0.0.34 (2025-04-19 18:30:00))
- Version numbers go in file content, NOT in filenames (we use git for versioning)
- When loading notebooks in Colab, always load from the current branch, not main
- Only switch back to main branch after PR is merged 
- Always test notebook functionality in both CPU and GPU (T4) environments
- Always evaluate pruned model to define variables before fine-tuning
- When fixing colab notebooks, create the corresponding .ipynb from the .py file
- Always create unit tests for new modules and classes
- Modularize visualizations and utilities into reusable components
- Add type hints to all function signatures for better code quality
- Keep notebook outputs clean using persistent display widgets
- Use consistent visualization styles across all notebooks

## Notebook Editing Protocol
- NEVER use sed, awk, or regex tools to edit .ipynb files
- Use the official Python SDK nbformat for all modifications to code cells, metadata, or outputs:
  ```python
  import nbformat
  
  nb = nbformat.read("notebooks/my_notebook.ipynb", as_version=4)
  for cell in nb.cells:
      if cell.cell_type == "code":
          cell.source = cell.source.replace("old_fn()", "new_fn()")
  nbformat.write(nb, "notebooks/my_notebook_fixed.ipynb")
  ```
- Jupyter notebooks are structured JSON and fragile - avoid regex-based edits
- Always run `python notebooks/validate_notebook.py notebook_path` after making changes
- Use jupytext to convert .py → .ipynb when working with script-first notebooks
- Move all notebook code to utility functions in appropriate modules (use `utils/colab/` for visualization)
- All notebook components (graphs, widgets, etc.) must be modular and reusable
- All notebooks must have corresponding unit tests before submission
- Unit test all entry points before each commit and before finishing a PR
- Test all notebooks in both CPU and GPU (T4) Colab environments before submitting
- Validate all visualizations and graphs manually before submission
- Unit test notebook outputs where applicable (numerical values, metrics, text outputs)
- Create test fixtures for visualization components to verify correct rendering
- Use `.private` directory for sharing test data and screenshots from Colab runs
- Compare visualization outputs with expected reference images stored in `.private`
- Always check timestamps on reference images - old data may be outdated
- Update reference images when visualization changes are expected and intentional
- Every feature must work and be fully tested at all times - no exceptions

## Build Commands
- Train model: `python train.py --model_name MODEL --dataset DATASET --epochs N --batch_size B --lr LR --device {cpu,cuda}`
- Run inference: `python main.py --model_name MODEL --prompt "text" --max_length L --device {cpu,cuda} --temperature T`
- Validate notebook: `python notebooks/validate_notebook.py notebook_path`
- Fix notebook metadata: `python notebooks/fix_notebook_metadata.py`
- Test pruning: `python scripts/inference_with_pruning.py --strategy entropy --pruning_level 0.5 --prompt "Your text"`
- Test improved fine-tuner: `python scripts/test_improved_fine_tuner.py --model MODEL --strategy entropy --pruning_level 0.3 --epochs 2`
- Test model support: `python test_model_support.py --device {cpu,cuda} --verbose`
- Run unit tests: `pytest` or `python -m pytest`
- Run specific test: `pytest utils/pruning/api/tests/test_pruning_impl.py -v`
- Run all pruning tests: `./scripts/run_pruning_tests.sh`
- Run standalone pruning test: `python tests/run_all_pruning_tests.py`
- Run comprehensive benchmark: `python scripts/benchmark_with_metrics.py --model_name distilgpt2 --eval_dataset gutenberg --use_real_data`

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
- Testing: Write unit tests for all new modules and functions
- Type hints: Add proper type annotations to all function signatures
- Docstrings: Use consistent docstring format with Args, Returns, and Raises sections

## Modularity Guidelines
- Refactor common functionality into separate utility modules
- Create classes for related functionality rather than using loose functions
- Prefer composition over inheritance for flexibility
- Make visualization code independent of model code
- Factor out data processing logic from training loops
- Break large functions into smaller, testable components
- Keep notebook-specific code minimal by importing from modules

## Project Structure
- Models in `models/`, with model loaders for different architectures in `models/loaders/`
- Controller logic in `controller/`, utilities in `utils/`
- Main training logic in `train.py`, inference in `main.py`
- Notebooks in `notebooks/` demonstrate various model behaviors and visualizations
- Scripts in `scripts/` provide benchmarking, fine-tuning, and specialized tools
- Documentation in `docs/` includes methodology, fine-tuning guides, and technical details
- Visualization utilities in `utils/colab/` for interactive displays and dashboards
- Unit tests in `tests/unit/` organized to mirror the project structure

## Visualization Standards
- Use persistent display widgets from `utils/colab/visualizations.py` for all notebooks
- Prefer updating displays in-place rather than creating new output cells
- Standardize on common color schemes for metrics (red for loss, green for accuracy)
- Use consistent figure sizes and layouts across notebooks
- Add proper axis labels, titles, and colorbars to all visualizations
- Include captions explaining the significance of key visualizations
- Set reasonable y-limits on plots to avoid excessive scaling

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
5. Increment version number when modifying dependencies or making significant changes to functionality.

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

## Testing Guidelines
- Create unit tests for all new functionality
- Organize tests to mirror the module structure (`tests/unit/module_name/test_file.py`)
- Test both happy paths and error cases
- Use parameterized tests for checking multiple inputs
- Mock external dependencies (models, datasets) for faster testing
- Include tests for edge cases and numerical stability
- Run tests in CI for both CPU and GPU environments

## Lessons Learned
- Use persistent widgets instead of multiple output cells in notebooks
- Test visualization code separately from model code
- Ensure proper error handling for common API failures
- Use dedicated visualization modules for consistency
- Ensure matplotlib figures have proper size limits set
- Prefer reusable utility classes over inline code
- Batch tool calls with BatchTool when processing multiple files
- Keep notebook outputs clean with in-place updates
- Test model loading with various init parameters like `from_tf`
- When starting a new session, first examine README.md, paper docs, and CLAUDE.md to understand context
- Read all relevant code thoroughly before making changes to understand dependencies and patterns
- Review the entire notebook before making edits to understand the full flow and dependencies
- ALWAYS activate the virtual environment with `source .venv/bin/activate` before running scripts
- ALWAYS use virtual environment for testing code: `source .venv/bin/activate && python your_script.py`
- When running scripts or tests, ALWAYS change directory to project root first: `cd /path/to/sentinel-ai && source .venv/bin/activate && python scripts/your_script.py`
- Always run notebooks end-to-end even in CPU mode before committing changes
- Always check tensor visualization code to ensure proper .detach().cpu().numpy() conversion
- Verify all files are properly committed (git status) before ending a session
- Test notebooks in Colab to ensure they work in both CPU and GPU environments
- Prefer standard GPT-2 over DistilGPT-2 to avoid potential crashes
- Always manage memory carefully with strategic `del` and `gc.collect()` calls
- Clean up disk space by removing old models and large data files when no longer needed
- Close and release resources like matplotlib figures when done with visualizations
- Monitor disk space usage and clean up temporary files regularly
- Use smaller/sampled datasets for testing and demonstrations
- Always check system date (currently 2025) before adding timestamps
- Use safe_tensor_imshow for GPU tensor visualization instead of raw plt.imshow
- Never make assumptions about the current date - get it from the system
- Carefully test code for syntax errors before committing
- On Apple Silicon (M1/M2/M3), safeguard against BLAS crashes by moving tensors to CPU for matrix operations
- Always include architecture-specific optimizations for Apple Silicon to avoid libtorch/BLAS crashes
- In Colab with GPU (T4), optimize for CUDA acceleration and ensure tensors are on GPU
- Use platform and hardware detection at module import time to automatically adjust execution
- Design code to be universally compatible with Apple Silicon, Colab T4 GPU, and standard CPU/GPU
- Always test tensor operations across all environments before finalizing code
- For visualization functions, always handle GPU tensors with proper detach+cpu conversion
- Provide fallback mechanisms for tensor operations to ensure robustness across platforms
