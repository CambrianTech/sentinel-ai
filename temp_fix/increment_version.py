import json
import os

# Load the notebook
notebook_path = '/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Find the title cell and update the version
title_cell = notebook['cells'][0]
content = title_cell['source']

# Find and replace the version in the title
if content.startswith('# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.16)'):
    content = content.replace('(v0.0.16)', '(v0.0.17)')
    title_cell['source'] = content
    
    # Add new version info
    new_version_text = "\n### New in v0.0.17:\n- Fixed visualization scaling to prevent extremely large plots\n- Added data downsampling for large training runs\n- Set explicit DPI control to maintain reasonable image sizes\n- Improved epoch boundary visualization\n"
    # Find position after first version block
    version_blocks = content.split('### New in v')
    if len(version_blocks) > 1:
        # Insert after title and before the first version block
        title_part = version_blocks[0]
        rest_part = '### New in v' + '### New in v'.join(version_blocks[1:])
        new_content = title_part + new_version_text + rest_part
        title_cell['source'] = new_content

# Update the conclusion cell with version history
for cell in notebook['cells']:
    if 'cell_type' in cell and cell['cell_type'] == 'markdown':
        content = ''.join(cell['source'])
        if '## Version History' in content:
            lines = content.split('\n')
            version_history_index = lines.index('## Version History')
            
            # Add new version info after the version history heading
            new_lines = lines[:version_history_index+1] + [
                '',
                '- v0.0.17: Fixed visualization scaling to prevent extremely large plots,',
                '           added data downsampling for training history, improved epoch visualization'
            ] + lines[version_history_index+1:]
            
            cell['source'] = '\n'.join(new_lines)
            break

# Save the modified notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Updated version in {notebook_path} to v0.0.17")