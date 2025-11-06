#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Package Pruning Comparison for Colab

This script:
1. Creates a packaged version of the pruning comparison script with all dependencies
2. Resolves import errors by including the correct class imports
3. Creates a convenient download link for Google Colab

Usage:
    python package_for_colab.py
"""

import os
import sys
import re
from pathlib import Path
import shutil
import tempfile
import zipfile

def main():
    """Main function to package pruning comparison for Colab."""
    print("Packaging pruning comparison for Google Colab...")
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent.absolute()
    os.chdir(project_root)
    
    # Create a temporary directory for packaging
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create directories structure
        for dir_path in ["scripts/pruning_comparison", "models", "models/loaders", "utils"]:
            os.makedirs(temp_path / dir_path, exist_ok=True)
        
        # Copy necessary script files
        shutil.copy("scripts/pruning_comparison/pruning_agency_comparison.py", 
                   temp_path / "scripts/pruning_comparison/")
        shutil.copy("scripts/pruning_comparison/run_pruning_comparison_colab.py", 
                   temp_path / "scripts/pruning_comparison/")
        
        # Copy model files
        shutil.copy("models/adaptive_transformer.py", temp_path / "models/")
        shutil.copy("models/loaders/loader.py", temp_path / "models/loaders/")
        shutil.copy("models/loaders/gpt2_loader.py", temp_path / "models/loaders/")
        
        # Copy utility files
        shutil.copy("utils/metrics.py", temp_path / "utils/")
        shutil.copy("utils/charting.py", temp_path / "utils/")
        
        # Create __init__.py files for imports to work
        for dir_path in ["models", "models/loaders", "utils"]:
            with open(temp_path / dir_path / "__init__.py", "w") as f:
                f.write("# Package initialization file\n")
        
        # Fix import in pruning_agency_comparison.py
        fix_import(temp_path / "scripts/pruning_comparison/pruning_agency_comparison.py")
        
        # Create a README with instructions
        create_readme(temp_path)
        
        # Create a ZIP file with the contents
        output_dir = project_root / "dist"
        os.makedirs(output_dir, exist_ok=True)
        
        zip_path = output_dir / "pruning_comparison_colab.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_path)
                    zipf.write(file_path, arcname)
        
        print(f"Package created successfully at: {zip_path}")
        print("\nInstructions for Google Colab:")
        print("1. Upload the ZIP file to your Google Drive")
        print("2. In Colab, use the following code to extract and run:")
        print("   !unzip /content/drive/MyDrive/pruning_comparison_colab.zip -d /content/")
        print("   !python /content/scripts/pruning_comparison/run_pruning_comparison_colab.py")

def fix_import(file_path):
    """Fix import statements in the file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add necessary import
    content = re.sub(
        r'from models.loaders.loader import load_baseline_model, load_adaptive_model',
        'from models.loaders.loader import load_baseline_model, load_adaptive_model\n'
        'from models.adaptive_transformer import AdaptiveCausalLmWrapper  # Import correct model class',
        content
    )
    
    # Fix other imports if needed
    content = re.sub(
        r'import sys\nimport os',
        'import sys\nimport os\n\n# Add parent directory to path\nsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))',
        content
    )
    
    # Add error handling for the execution part
    content = re.sub(
        r'start_time = time.time\(\)\n!{" ".join\(cmd\)}\nend_time = time.time\(\)',
        'start_time = time.time()\ntry:\n    !{" ".join(cmd)}\n    experiment_success = True\n'
        'except Exception as e:\n    print(f"Error during experiment: {str(e)}")\n    experiment_success = False\n'
        'end_time = time.time()',
        content
    )
    
    # Add automatic patching for AdaptiveCausalLmWrapper
    if 'run_pruning_comparison_colab.py' in str(file_path):
        content = re.sub(
            r'# Install dependencies\nprint\("Installing dependencies..."\)',
            '# Fix the import error: Create a Python script to patch the imports\n'
            'print("Patching imports for compatibility...")\n'
            'with open(\'fix_imports.py\', \'w\') as f:\n'
            '    f.write("""\nimport os\nimport re\n\n'
            '# Files to patch\nfiles_to_patch = [\n'
            '    \'scripts/pruning_comparison/pruning_agency_comparison.py\',\n]\n\n'
            'for file_path in files_to_patch:\n'
            '    if not os.path.exists(file_path):\n'
            '        print(f"Warning: File {file_path} not found")\n'
            '        continue\n'
            '        \n'
            '    # Read the file\n'
            '    with open(file_path, \'r\') as f:\n'
            '        content = f.read()\n'
            '    \n'
            '    # Add necessary import if missing\n'
            '    if \'from models.adaptive_transformer import AdaptiveCausalLmWrapper\' not in content:\n'
            '        content = re.sub(\n'
            '            r\'from models.loaders.loader import load_baseline_model, load_adaptive_model\',\n'
            '            \'from models.loaders.loader import load_baseline_model, load_adaptive_model\\\\n\'\n'
            '            \'from models.adaptive_transformer import AdaptiveCausalLmWrapper  # Import correct model class\',\n'
            '            content\n'
            '        )\n'
            '        \n'
            '        # Write the modified content back\n'
            '        with open(file_path, \'w\') as f:\n'
            '            f.write(content)\n'
            '        print(f"✓ Fixed imports in {file_path}")\n'
            '    else:\n'
            '        print(f"✓ File {file_path} already has correct imports")\n""")\n\n'
            '# Run the import fixer\n!python fix_imports.py\n\n'
            '# Install dependencies\nprint("Installing dependencies...")',
            content
        )
    
    with open(file_path, 'w') as f:
        f.write(content)

def create_readme(temp_path):
    """Create a README file with instructions."""
    readme_content = """# Pruning Comparison for Google Colab

This package contains the pruning comparison framework for running on Google Colab with a T4 GPU.

## Quick Start

1. Run the Colab script directly:
   ```python
   !python /content/scripts/pruning_comparison/run_pruning_comparison_colab.py
   ```

2. This will:
   - Install required dependencies
   - Run the pruning comparison experiment
   - Generate visualizations
   - Save results to Google Drive (if mounted)

## Files Included

- `scripts/pruning_comparison/run_pruning_comparison_colab.py`: Main Colab runner script
- `scripts/pruning_comparison/pruning_agency_comparison.py`: Core comparison implementation
- `models/adaptive_transformer.py`: Adaptive transformer model with agency mechanisms
- `models/loaders/`: Model loading utilities
- `utils/`: Metrics and visualization utilities

## Customization

You can modify parameters in the Colab script to customize the experiment:

```python
model_name = "gpt2"  # or "distilgpt2"
pruning_levels = "0,20,40,60,80"  # Comma-separated pruning percentages
num_tokens = 100     # Tokens to generate per prompt
temperatures = "0.7,1.0"  # Temperatures to test
iterations = 3       # Statistical significance iterations
```

## Results

Results will be automatically saved to:
- `/content/validation_results/pruning_agency/latest/`
- Google Drive if mounted (under a timestamped folder)

Generated visualizations include:
- Speed and quality comparison charts
- Relative improvement metrics
- Comprehensive summary visuals
"""
    
    with open(temp_path / "README.md", "w") as f:
        f.write(readme_content)

if __name__ == "__main__":
    main()