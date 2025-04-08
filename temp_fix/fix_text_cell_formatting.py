#!/usr/bin/env python
# Comprehensive fix for text cell formatting issues

import json
import re
from pathlib import Path

def fix_text_cell_formatting(notebook_path):
    """Thoroughly fix all text cell formatting issues."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Process all markdown cells
        fixed_cells = 0
        
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'markdown':
                # Get current content and join it to see the full text
                current_content = ''.join(cell['source'])
                
                # Print a sample of problematic cells for debugging
                if i < 5 or '# Define Evaluation' in current_content:
                    print(f"\nCell {i} original content (snippet):")
                    preview = repr(current_content[:100] + ('...' if len(current_content) > 100 else ''))
                    print(preview)
                
                # Look for specific formatting issues
                formatting_issues = False
                
                # Issue 1: Headers without proper spacing
                if re.search(r'#[^#\n]*[a-zA-Z]\n[a-zA-Z]', current_content):
                    formatting_issues = True
                
                # Issue 2: Header with split lines
                if re.search(r'# \w+.*\n\w+', current_content):
                    formatting_issues = True
                
                # Issue 3: Text with random newlines
                if re.search(r'[a-z]\n[a-z]', current_content):
                    formatting_issues = True
                
                # Apply comprehensive fixes
                if formatting_issues or i < 10 or '# Define Evaluation' in current_content:
                    # Get the raw lines to see exactly how it's broken
                    raw_lines = cell['source']
                    print(f"Cell {i} raw lines: {raw_lines}")
                    
                    # APPROACH 1: Complete rebuild of markdown content
                    # Extract headers
                    headers = re.findall(r'(#+\s+[^#\n]+)', current_content)
                    
                    # Extract paragraphs - any text that's not a header
                    text_content = re.sub(r'#+\s+[^#\n]+', '\n\n', current_content)
                    
                    # Pull out paragraphs
                    paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
                    
                    # Remove extraneous newlines within paragraphs
                    clean_paragraphs = []
                    for p in paragraphs:
                        # Preserve lists and code blocks
                        if p.startswith('- ') or p.startswith('1. ') or p.startswith('```'):
                            clean_paragraphs.append(p)
                        else:
                            # Join text with random newlines
                            cleaned = re.sub(r'\s*\n\s*', ' ', p)
                            clean_paragraphs.append(cleaned)
                    
                    # APPROACH 2: Define specific fixes for problematic cells
                    if '# Define Evaluation' in current_content:
                        # Fix the specific evaluation function cell
                        fixed_content = "# Define Evaluation Function\n\nLet's define a function to evaluate our model's performance\n"
                        cell['source'] = fixed_content.split('\n')
                        fixed_cells += 1
                        print(f"Fixed 'Define Evaluation' cell {i}")
                        continue
                        
                    # Fix title cell
                    if i == 0:
                        fixed_content = "# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.30)\n\nThis notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to dynamically prune and regrow attention heads during training based on utility metrics.\n\n## What is Neural Plasticity?\n\nNeural plasticity is the ability of neural networks to adapt their structure over time through pruning (removing unused connections) and regrowth (restoring useful connections). This mimics how biological brains form efficient neural pathways.\n\nIn this demo, we:\n1. Track the entropy and gradient patterns of each attention head\n2. Dynamically prune high-entropy, low-gradient heads (unfocused, less useful)\n3. Selectively revive low-entropy, higher-gradient heads (potentially useful)\n4. Visualize the \"brain dynamics\" over time\n\nThis allows models to form more efficient neural structures during training.\n\n### New in v0.0.30:\n- Fixed markdown formatting throughout notebook\n- Improved newline handling in all sections\n- Fixed spacing between paragraphs and headers\n- Enhanced list formatting and readability\n- Fixed title formatting issues\n\n### New in v0.0.29:\n- Fixed visualization issues and rendering in matplotlib\n- Improved training metrics visualization display\n- Added better entropy visualization with non-zero ranges\n- Fixed constrained_layout warnings\n\n### New in v0.0.28:\n- Fixed entropy calculation to prevent zero values\n- Added improved visualization of attention patterns\n- Fixed graph rendering issues in matplotlib\n- Added direct calculation of entropy from attention values\n\n### New in v0.0.27:\n- Attempt at fixing an infinitely long graph\n\n### New in v0.0.25:\n- Fixed layout issues\n\n### New in v0.0.23:\n- Fixed visualization issues causing excessively large images\n- Reduced figure sizes and DPI settings\n- Fixed cell splitting in controller section"
                        cell['source'] = fixed_content.split('\n')
                        fixed_cells += 1
                        print(f"Fixed title cell {i}")
                        continue
                    
                    # Fix What is Neural Plasticity cell
                    if 'What is Neural Plasticity' in current_content and len(current_content) < 500:
                        fixed_content = "## What is Neural Plasticity?\n\nNeural plasticity is the ability of neural networks to adapt their structure over time through pruning (removing unused connections) and regrowth (restoring useful connections). This mimics how biological brains form efficient neural pathways.\n\nIn this demo, we implement a neural plasticity system that allows a model to dynamically adapt during training by:\n\n1. Pruning attention heads with high entropy (unfocused) and low gradient impact\n2. Potentially reviving heads that show renewed importance\n3. Visualizing the changes in neural structure over time"
                        cell['source'] = fixed_content.split('\n')
                        fixed_cells += 1
                        print(f"Fixed Neural Plasticity cell {i}")
                        continue
                    
                    # General fix for all other cells - rebuild with proper newlines
                    # Convert all content to a single string and fix known issues
                    content = ''.join(cell['source'])
                    
                    # Fix 1: Headers should have \n\n after them
                    content = re.sub(r'(#+[^#\n]+)(\n[^#\n])', r'\1\n\n\2', content)
                    
                    # Fix 2: Remove random newlines in paragraphs
                    # This is tricky - we need to preserve intentional newlines like lists
                    paragraphs = re.split(r'\n\s*\n', content)
                    fixed_paragraphs = []
                    
                    for p in paragraphs:
                        # Skip empty paragraphs
                        if not p.strip():
                            continue
                            
                        # If it's a header, preserve as is
                        if p.strip().startswith('#'):
                            fixed_paragraphs.append(p)
                            continue
                            
                        # If it's a list, preserve as is
                        if p.strip().startswith('- ') or p.strip().startswith('1. '):
                            fixed_paragraphs.append(p)
                            continue
                            
                        # Otherwise, fix random newlines within paragraphs
                        fixed_p = re.sub(r'([a-z])\s*\n\s*([a-z])', r'\1 \2', p)
                        fixed_paragraphs.append(fixed_p)
                    
                    # Join back with double newlines
                    fixed_content = '\n\n'.join(fixed_paragraphs)
                    
                    # Apply the fix
                    cell['source'] = [fixed_content]
                    fixed_cells += 1
                    print(f"Applied general formatting fix to cell {i}")
        
        # Save the notebook
        if fixed_cells > 0:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1)
            print(f"Fixed {fixed_cells} text cells")
            return True
        else:
            print("No cells needed fixing")
            return False
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_text_cell_formatting(notebook_path)