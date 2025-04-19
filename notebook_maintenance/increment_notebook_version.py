#!/usr/bin/env python
"""
Increment version number in NeuralPlasticityDemo.ipynb

This script increments the version number in the notebook header
to indicate the bug fixes applied.
"""

import nbformat
import sys
import re
import datetime

def increment_version(notebook_path, output_path=None, include_timestamp=True):
    """
    Increment the version number in the notebook header
    
    Args:
        notebook_path: Path to the notebook
        output_path: Optional path to save the modified notebook
    """
    print(f"Reading notebook: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    
    # Find the first markdown cell with a title containing version number
    title_cell = None
    version_pattern = r"v(\d+)\.(\d+)\.(\d+)"
    current_version = None
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and "# Neural Plasticity Demo" in cell.source:
            title_cell = cell
            # Extract version
            match = re.search(version_pattern, cell.source)
            if match:
                current_version = tuple(map(int, match.groups()))
            break
    
    if not title_cell or not current_version:
        print("Could not find version information in notebook header")
        return False
    
    # Increment version (increment patch number)
    major, minor, patch = current_version
    new_version = (major, minor, patch + 1)
    
    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_version_str = f"v{new_version[0]}.{new_version[1]}.{new_version[2]} ({timestamp})"
    else:
        new_version_str = f"v{new_version[0]}.{new_version[1]}.{new_version[2]}"
    
    # Extract the version number without timestamp for comparison
    base_version_str = f"v{new_version[0]}.{new_version[1]}.{new_version[2]}"
    
    # Update version in the title cell - handle both with and without timestamp
    if " (" in title_cell.source:
        # There's already a timestamp, replace the whole thing
        title_cell.source = re.sub(
            r"v\d+\.\d+\.\d+\s+\([^)]+\)",
            new_version_str,
            title_cell.source
        )
    else:
        # No timestamp yet, replace just the version number
        title_cell.source = re.sub(
            version_pattern,
            new_version_str,
            title_cell.source
        )
    
    # Add a note about the fixes to the first markdown cell
    # Check if we're already adding version notes or need to add a new section
    if f"### New in {base_version_str}:" in title_cell.source:
        # The version has been already updated, so just append new changes
        bullet_points = [
            "- Added %matplotlib inline for Colab compatibility",
            "- Added system dependency checks",
            "- Improved error handling in training loop",
            "- Deduplicated import statements",
            "- Fixed cell execution counts for better notebook flow"
        ]
        for point in bullet_points:
            if point not in title_cell.source:
                title_cell.source += f"\n{point}"
    else:
        # Add a completely new version section
        title_cell.source += f"\n\n### New in {base_version_str}:\n- Fixed GPU tensor visualization errors\n- Fixed visualization utilities integration\n- Ensured proper tensor detachment and CPU conversion for visualization\n- Integrated with utils.colab.visualizations module\n- Added %matplotlib inline for Colab compatibility\n- Added system dependency checks\n- Improved error handling in training loop\n- Deduplicated import statements\n- Fixed cell execution counts for better notebook flow"
    
    # Save the updated notebook
    output_file = output_path or notebook_path
    print(f"Saving notebook with updated version to: {output_file}")
    nbformat.write(nb, output_file)
    
    # Return values for display
    old_version = f"v{major}.{minor}.{patch}"
    
    print(f"Version updated from {old_version} to {new_version_str}")
    return {"old_version": old_version, "new_version": new_version_str, "base_version": base_version_str}


if __name__ == "__main__":
    # Get notebook path from command line or use default
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else "colab_notebooks/NeuralPlasticityDemo.ipynb"
    
    # Option to specify output path as second argument
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Option to disable timestamp (for testing)
    include_timestamp = True
    if len(sys.argv) > 3 and sys.argv[3].lower() == "notimestamp":
        include_timestamp = False
    
    # Increment version
    increment_version(notebook_path, output_path, include_timestamp)