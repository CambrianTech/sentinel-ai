import nbformat
import sys

filename = sys.argv[1]

try:
    with open(filename, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    print(f"{filename}: ✅ Notebook format is valid.")
except Exception as e:
    print(f"{filename}: ❌ Error detected!")
    print(f"Error details: {e}")
