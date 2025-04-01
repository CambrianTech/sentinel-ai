# CLAUDE.md - AI Agent Guidelines

## Build Commands
- Train model: `python train.py --model_name MODEL --dataset DATASET --epochs N --batch_size B --lr LR --device {cpu,cuda}`
- Run inference: `python main.py --model_name MODEL --prompt "text" --max_length L --device {cpu,cuda} --temperature T`
- Validate notebook: `python notebooks/validate_notebook.py notebook_path`
- Fix notebook metadata: `python notebooks/fix_notebook_metadata.py`

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