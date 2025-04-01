# CLAUDE.md - AI Agent Guidelines

## Build Commands
- Train model: `python train.py --model_name MODEL --dataset DATASET --epochs N --batch_size B --lr LR --device {cpu,cuda}`
- Run inference: `python main.py --model_name MODEL --prompt "text" --max_length L --device {cpu,cuda} --temperature T`
- Validate notebook: `python notebooks/validate_notebook.py notebook_path`
- Fix notebook metadata: `python notebooks/fix_notebook_metadata.py`
- Test pruning: `python scripts/inference_with_pruning.py --strategy entropy --pruning_level 0.5 --prompt "Your text"`

## Code Style Guidelines
- Imports: standard library → third-party → local modules
- Naming: snake_case for functions/variables, CamelCase for classes, UPPER_CASE for constants
- Error handling: Use try/except with specific exceptions and meaningful error messages
- Parameters: Provide sensible defaults where appropriate
- Documentation: Add comments for complex logic, use argparse with helpful descriptions
- Organization: Keep functions single-purpose and modular

## Project Structure
- Models in `models/`, controller logic in `controller/`, utilities in `utils/`
- Main training logic in `train.py`, inference in `main.py`
- Notebooks in `notebooks/` demonstrate various model behaviors and visualizations
- Scripts in `scripts/` provide benchmarking and specialized tools
- Documentation in `docs/` includes methodology and technical details

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