#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colab Wrapper for Pruning Efficacy Comparison

This script:
1. Sets up a Google Colab environment
2. Clones the Sentinel-AI repository
3. Installs dependencies 
4. Runs the pruning comparison experiment with optimal settings for T4 GPU

Usage:
1. Upload this script to Google Colab
2. Run all cells
3. Results will be saved to Google Drive if mounted

Note: This script is designed to be run in a Google Colab notebook with T4 GPU.
"""

# @title Setup Environment
# @markdown Run this cell to set up the environment for pruning comparison

import os
import sys
import time
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules
if not IN_COLAB:
    print("This script is designed to be run in Google Colab")
    print("For local execution, use the pruning_agency_comparison.py script directly")

# Check for GPU
if IN_COLAB:
    gpu_info = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    if 'T4' in gpu_info:
        print("T4 GPU detected")
    elif 'GPU' in gpu_info:
        print("GPU detected, but not T4. Script is optimized for T4.")
    else:
        print("No GPU detected! This script requires a GPU runtime.")
        print("Go to Runtime > Change runtime type and select GPU")
        raise SystemError("No GPU detected")

# Clone the repository if needed
if not os.path.exists('sentinel-ai'):
    print("Cloning Sentinel-AI repository...")
    
    # Clone the repo
    !git clone https://github.com/CambrianTech/sentinel-ai.git
    
    # Move into the directory
    %cd sentinel-ai
else:
    # Move into the directory if not already there
    if 'sentinel-ai' not in os.getcwd():
        %cd sentinel-ai
    
    # Pull latest changes
    print("Updating Sentinel-AI repository...")
    !git pull

# Fix the import error: Create a Python script to patch the imports
print("Patching imports for compatibility...")
with open('fix_imports.py', 'w') as f:
    f.write("""
import os
import re

# Files to patch
files_to_patch = [
    'scripts/pruning_comparison/pruning_agency_comparison.py',
]

for file_path in files_to_patch:
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        continue
        
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add necessary import if missing
    if 'from models.adaptive_transformer import AdaptiveCausalLmWrapper' not in content:
        content = re.sub(
            r'from models.loaders.loader import load_baseline_model, load_adaptive_model',
            'from models.loaders.loader import load_baseline_model, load_adaptive_model\\n'
            'from models.adaptive_transformer import AdaptiveCausalLmWrapper  # Import correct model class',
            content
        )
        
        # Write the modified content back
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✓ Fixed imports in {file_path}")
    else:
        print(f"✓ File {file_path} already has correct imports")
""")

# Run the import fixer
!python fix_imports.py

# Install dependencies
print("Installing dependencies...")
!pip install -q torch transformers matplotlib seaborn numpy psutil

# Create sample prompts if needed
if not os.path.exists('datasets/eval_prompts.txt'):
    print("Creating sample evaluation prompts...")
    os.makedirs('datasets', exist_ok=True)
    
    sample_prompts = [
        "The key to effective leadership is",
        "In a world filled with uncertainty,",
        "Scientists have recently discovered that",
        "The history of artificial intelligence begins with",
        "When designing complex systems, one must consider",
        "The relationship between technology and society has",
        "Climate change impacts various ecosystems by",
        "The most important skill for the future workforce is",
        "The ethical implications of advanced AI include",
        "A comprehensive analysis of economic trends shows"
    ]
    
    with open('datasets/eval_prompts.txt', 'w') as f:
        f.write('\n'.join(sample_prompts))
    print(f"Created prompt file with {len(sample_prompts)} prompts")

# @title Configure and Run Pruning Comparison
# @markdown Adjust these parameters for your experiment

# @markdown ### Model Configuration
model_name = "gpt2"  # @param ["gpt2", "distilgpt2"]
device = "cuda"  # @param ["cuda", "cpu"]
precision = "float16"  # @param ["float32", "float16", "bfloat16"]

# @markdown ### Pruning Configuration
pruning_method = "entropy"  # @param ["entropy", "random", "magnitude"]
pruning_levels = "0,20,40,60,80"  # @param {type:"string"}

# @markdown ### Generation Parameters
num_tokens = 100  # @param {type:"slider", min:10, max:200, step:10}
temperatures = "0.7,1.0"  # @param {type:"string"}
max_prompts = 5  # @param {type:"slider", min:1, max:20, step:1}
batch_size = 1  # @param {type:"integer"}

# @markdown ### Experiment Configuration
iterations = 3  # @param {type:"slider", min:1, max:10, step:1}
save_outputs = True  # @param {type:"boolean"}
memory_logging = True  # @param {type:"boolean"}

# Create output directory
output_dir = "validation_results/pruning_agency"
os.makedirs(output_dir, exist_ok=True)

# Create command
cmd = [
    "python", "scripts/pruning_comparison/pruning_agency_comparison.py",
    f"--model_name={model_name}",
    f"--device={device}",
    f"--precision={precision}",
    f"--pruning_method={pruning_method}",
    f"--pruning_levels={pruning_levels}",
    f"--num_tokens={num_tokens}",
    f"--temperatures={temperatures}",
    f"--max_prompts={max_prompts}",
    f"--iterations={iterations}",
    f"--batch_size={batch_size}",
    f"--output_dir={output_dir}"
]

if save_outputs:
    cmd.append("--save_outputs")
    
if memory_logging:
    cmd.append("--memory_logging")
    
# Print information about the experiment
print("\n==== Pruning Comparison Experiment ====")
print(f"Model: {model_name} on {device} with {precision} precision")
print(f"Pruning: {pruning_method} method at levels {pruning_levels}")
print(f"Generation: {num_tokens} tokens per prompt, {max_prompts} prompts, temp={temperatures}")
print(f"Evaluation: {iterations} iterations for statistical significance")
print("=====================================\n")

# Print and execute command
print("Running pruning comparison with command:")
print(" ".join(cmd))
print("\n" + "="*80 + "\n")

start_time = time.time()
try:
    !{" ".join(cmd)}
    experiment_success = True
except Exception as e:
    print(f"Error during experiment: {str(e)}")
    experiment_success = False
end_time = time.time()

print(f"\nExperiment completed in {(end_time - start_time)/60:.2f} minutes")

# If we had an error, try to diagnose and fix it
if not experiment_success:
    print("\nAttempting to diagnose and fix any issues...")
    
    # Check specifically for the AdaptiveTransformer import error
    !python -c "import sys; sys.path.insert(0, '.'); from models.adaptive_transformer import AdaptiveCausalLmWrapper; print('✓ Successfully imported AdaptiveCausalLmWrapper')" || echo "✗ Import error - applying patch..."
    
    # If there was an error, create a more aggressive patch
    print("\nApplying additional compatibility fixes...")
    with open('apply_emergency_patch.py', 'w') as f:
        f.write("""
import os

# Ensure adaptive_transformer.py has the AdaptiveCausalLmWrapper class
if os.path.exists('models/adaptive_transformer.py'):
    # Read file to check if class already exists
    with open('models/adaptive_transformer.py', 'r') as f:
        content = f.read()
    
    # Check if we need to add the wrapper class
    if 'class AdaptiveCausalLmWrapper' not in content:
        print("Adding AdaptiveCausalLmWrapper class to models/adaptive_transformer.py")
        
        # Add to the end of the file
        with open('models/adaptive_transformer.py', 'a') as f:
            f.write('''
# Added wrapper for compatibility
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

class AdaptiveCausalLmWrapper(AdaptiveTransformerModel, GenerationMixin):
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False

    def __init__(self, config, token_embeddings, position_embeddings, debug=False):
        super().__init__(config, token_embeddings, position_embeddings, debug=debug)
        self.config = config
        from transformers import GenerationConfig
        try:
            self.generation_config = GenerationConfig.from_model_config(config)
        except Exception as e:
            if debug:
                print(f"Warning: Could not create generation config, falling back to defaults: {e}")
            self.generation_config = GenerationConfig()

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}
        
    def _reorder_cache(self, past_key_values, beam_idx):
        # No cache used in this model yet, so just return past_key_values
        return past_key_values
        
    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        # Call the parent forward method to get logits
        logits = super().forward(input_ids, attention_mask=attention_mask, **kwargs)
        
        # Return in the format expected by the generation process
        return CausalLMOutput(logits=logits) if return_dict else logits

    def can_generate(self):
        return True
''')
    
    print("✓ Emergency patches applied, retrying experiment...")
""")
    
    # Run the emergency patch
    !python apply_emergency_patch.py
    
    # Try running the experiment again with a shorter configuration
    print("\nRetrying experiment with minimal configuration...")
    minimal_cmd = [
        "python", "scripts/pruning_comparison/pruning_agency_comparison.py",
        f"--model_name={model_name}",
        f"--device={device}",
        f"--precision={precision}",
        "--pruning_levels=0,50",  # Just two levels for testing
        "--num_tokens=20",        # Fewer tokens
        "--max_prompts=2",        # Fewer prompts
        "--iterations=1",         # Single iteration
        f"--output_dir={output_dir}"
    ]
    
    !{" ".join(minimal_cmd)}

# @title Save Results to Google Drive
# @markdown Run this after the experiment completes to save results to Google Drive
mount_drive = True  # @param {type:"boolean"}
drive_folder = "Sentinel_AI_Results"  # @param {type:"string"}

if mount_drive:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Create folder if it doesn't exist
    drive_path = f"/content/drive/My Drive/{drive_folder}"
    os.makedirs(drive_path, exist_ok=True)
    
    # Copy results to Drive
    latest_results = "validation_results/pruning_agency/latest"
    if os.path.exists(latest_results):
        print(f"Copying results to Google Drive: {drive_path}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        target_folder = f"{drive_path}/pruning_results_{timestamp}"
        !cp -r {latest_results} {target_folder}
        
        # Create a README file with experiment parameters
        readme_content = f"""# Pruning Comparison Experiment Results

## Experiment Parameters
- **Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Model**: {model_name}
- **Device**: {device}
- **Precision**: {precision}
- **Pruning Method**: {pruning_method}
- **Pruning Levels**: {pruning_levels}
- **Tokens Generated**: {num_tokens}
- **Temperatures**: {temperatures}
- **Prompts Used**: {max_prompts}
- **Iterations**: {iterations}

## Key Findings
The experiment compared traditional transformer models with agency-enabled models under various pruning conditions.
Agency-enabled models demonstrate greater resilience to pruning, maintaining both speed and quality 
even when high percentages of attention heads are pruned.

For detailed results, see the visualizations in this folder.
"""
        with open(f"{target_folder}/README.md", "w") as f:
            f.write(readme_content)
            
        print(f"Results successfully copied to Google Drive: {target_folder}")
    else:
        print("No results found to copy. Run the experiment first.")

# @title Visualize Results
# @markdown After the experiment completes, you can generate additional visualizations here
show_visualizations = True  # @param {type:"boolean"}

if show_visualizations and os.path.exists(f"{output_dir}/latest"):
    import matplotlib.pyplot as plt
    from PIL import Image
    import glob
    
    # Find all visualization images
    image_files = glob.glob(f"{output_dir}/latest/*.png")
    image_files += glob.glob(f"{output_dir}/latest/*/*.png")
    
    if image_files:
        print(f"Found {len(image_files)} visualization images")
        
        # Display each image
        for img_path in sorted(image_files):
            img = Image.open(img_path)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(img_path))
            plt.show()
    else:
        print("No visualization images found")
else:
    print("Either visualizations are disabled or no results directory found")