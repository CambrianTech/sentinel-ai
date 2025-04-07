import json
import os
import re

# Load the notebook
notebook_path = '/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Find all torch.topk usages in the notebook
topk_cells = []
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = ''.join(cell['source'])
        if 'torch.topk' in source:
            topk_cells.append((i, source))

print(f"Found {len(topk_cells)} cells with torch.topk usage")

# Check each cell for topk usage
for cell_num, source in topk_cells:
    print(f"\n=== Cell {cell_num} ===")
    
    # Extract lines with torch.topk
    lines = source.split('\n')
    topk_lines = [line for line in lines if 'torch.topk' in line]
    
    for line in topk_lines:
        print(f"topk usage: {line.strip()}")
        
        # Check for largest parameter
        if 'largest=False' in line:
            print("✓ Correctly targeting LOWEST values (largest=False)")
        elif 'largest=True' in line:
            print("✗ Targeting HIGHEST values (largest=True) - this may be incorrect")
        else:
            print("? No explicit largest parameter (defaults to largest=True)")
        
        # Check context to understand what's being selected
        context = []
        line_idx = lines.index(line)
        start_idx = max(0, line_idx - 5)
        end_idx = min(len(lines), line_idx + 5)
        
        # Get 5 lines before and after for context
        context = lines[start_idx:end_idx]
        
        # Look for comments or variable names that indicate intention
        for ctx_line in context:
            if 'LOWEST' in ctx_line:
                print("✓ Has comment indicating LOWEST values are targeted")
            if 'lowest' in ctx_line.lower() and not 'highest' in ctx_line.lower():
                print("✓ Has comment indicating lowest values are targeted")
            if 'highest' in ctx_line.lower() and not 'lowest' in ctx_line.lower():
                print("? Has comment indicating highest values might be targeted")

# Find the entropy calculation/visualization in the notebook
entropy_cells = []
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = ''.join(cell['source'])
        if 'entropy' in source.lower() and ('plt.imshow' in source or 'plt.plot' in source):
            entropy_cells.append((i, source))

print(f"\nFound {len(entropy_cells)} cells with entropy visualization")

# Check each cell for entropy visualization
for cell_num, source in entropy_cells:
    print(f"\n=== Cell {cell_num} ===")
    
    # Check if the cell is showing possibly zero entropy values
    if 'debug_entropy' in source and 'plt.imshow' in source:
        # Look for lines with entropy statistics
        lines = source.split('\n')
        stats_lines = [line for line in lines if 'entropy' in line.lower() and any(x in line for x in ['mean', 'min', 'max', 'percentile'])]
        
        for line in stats_lines:
            print(f"Entropy stat: {line.strip()}")
        
        # Check for specific entropy issues or debugging
        if 'all entropy values the same' in source.lower():
            print("✓ Has check for uniform entropy")
        if 'count_nonzero' in source and 'entropy' in source:
            print("✓ Has check for zero entropy values")
            
        # Extract any numerical entropy statistics
        mean_match = re.search(r'Mean entropy: ([0-9.]+)', source)
        min_match = re.search(r'Min entropy: ([0-9.]+)', source) 
        max_match = re.search(r'Max entropy: ([0-9.]+)', source)
        
        if mean_match and min_match and max_match:
            mean_val = float(mean_match.group(1))
            min_val = float(min_match.group(1))
            max_val = float(max_match.group(1))
            
            print(f"Found entropy values - Mean: {mean_val}, Min: {min_val}, Max: {max_val}")
            
            # Check if entropy values are likely zero
            if mean_val < 0.01 and max_val < 0.01:
                print("⚠️ Entropy values are very close to zero!")
            elif min_val < 0.01 and mean_val < 0.1:
                print("⚠️ Some entropy values are very close to zero!")
        
        # Look for visualizations
        viz_lines = [line for line in lines if 'plt.imshow' in line and 'entropy' in line.lower()]
        for line in viz_lines:
            print(f"Visualization: {line.strip()}")
            
            # Check for scaling parameters
            if 'vmin=0' in line:
                print("✓ Has minimum scale value")
            if 'vmax=' in line:
                print("✓ Has maximum scale value")
            else:
                print("? No maximum scale value specified")