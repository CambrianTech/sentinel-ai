import json
import os

# Load the notebook
notebook_path = '/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Check specific cells
for cell_num in [13, 15, 17]:
    print(f"\n=== Cell {cell_num} ===")
    if cell_num < len(notebook['cells']):
        cell = notebook['cells'][cell_num]
        if 'source' in cell:
            source = ''.join(cell['source'])
            if 'torch.topk' in source:
                print(f"Found torch.topk in cell {cell_num}")
                
                # Find lines containing torch.topk
                lines = cell['source']
                for i, line in enumerate(lines):
                    if 'torch.topk' in line:
                        print(f"  Line {i}: {line.strip()}")
                        # Check if using largest=False
                        if 'largest=False' in line:
                            print("  ✓ Correctly using largest=False to target lowest gradients")
                        elif 'largest=True' in line:
                            print("  ✗ Using largest=True which targets highest gradients!")
                        else:
                            print("  ? No explicit largest parameter, check default behavior")
            
            # Check for entropy calculations
            if 'entropy' in source.lower():
                print(f"Found entropy references in cell {cell_num}")
                # Look for issues with entropy calculation/visualization
                if 'entropy_values' in source and ('mean' in source or 'min' in source or 'max' in source):
                    print("  Contains entropy statistics")
                if 'are all entropy values the same' in source.lower():
                    print("  Has checks for zero/same entropy values")
            
            # Print a summary of the cell (first 200 chars)
            print(f"\nSummary: {source[:200]}...")
        else:
            print("No source in cell")
    else:
        print("Cell index out of bounds")