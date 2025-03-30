import json
import sys

file = sys.argv[1]

# Manually read JSON content
with open(file, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Ensure 'metadata' key exists
if 'metadata' not in notebook:
    notebook['metadata'] = {}

# Optionally, ensure each cell also has 'metadata'
for cell in notebook.get("cells", []):
    if 'metadata' not in cell:
        cell['metadata'] = {}

# Save fixed notebook
with open(file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"✅ Fixed metadata issue in {file}")
