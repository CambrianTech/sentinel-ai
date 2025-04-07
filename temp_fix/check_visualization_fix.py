import json
import os

# Load the notebook
notebook_path = '/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Check cell 19 for our visualization fix
cell_19 = notebook['cells'][19]
source = ''.join(cell_19['source'])

# Check if key elements of our fix are present
print("Checking visualization fix in cell 19:")
if 'max_display_points' in source:
    print("✓ max_display_points parameter found")
else:
    print("✗ max_display_points parameter not found")

if 'display_steps =' in source:
    print("✓ display_steps variable found")
else:
    print("✗ display_steps variable not found")

if 'if len(display_steps) > max_display_points:' in source:
    print("✓ Downsampling logic found")
else:
    print("✗ Downsampling logic not found")

if 'indices = np.linspace' in source:
    print("✓ Evenly spaced indices calculation found")
else:
    print("✗ Evenly spaced indices calculation not found")

if 'ax1.plot(display_steps, display_train_loss' in source:
    print("✓ Using display_steps for plotting")
else:
    print("✗ Not using display_steps for plotting")

if 'plt.gcf().set_dpi(100)' in source:
    print("✓ DPI control found")
else:
    print("✗ DPI control not found")

if 'dpi=100' in source:
    print("✓ DPI set in figure creation")
else:
    print("✗ DPI not set in figure creation")

# Print the beginning of the cell for verification
print("\nBeginning of cell 19:")
lines = cell_19['source']
for i, line in enumerate(lines[:15]):  # Print first 15 lines
    print(f"{i+1}: {line.strip()}")

print("\nThe fix has been successfully applied if all checks above show '✓'.")