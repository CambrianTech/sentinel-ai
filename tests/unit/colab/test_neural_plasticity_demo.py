"""
Tests for the NeuralPlasticityDemo notebook.

These tests verify that the Neural Plasticity Demo notebook can be loaded
and has the expected structure without actually running it.
"""

import os
import sys
import unittest
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestNeuralPlasticityNotebook(unittest.TestCase):
    """Test case for checking the Neural Plasticity Demo notebook."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.notebook_path = os.path.join(project_root, "colab_notebooks/NeuralPlasticityDemo.ipynb")
        
        # Check that the notebook exists
        self.assertTrue(
            os.path.exists(self.notebook_path),
            f"Notebook not found at {self.notebook_path}"
        )
        
        # Load the notebook
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            self.notebook = json.load(f)
        
        # Extract cell sources for easier testing
        self.cell_sources = []
        for cell in self.notebook['cells']:
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            self.cell_sources.append(source)
    
    def test_notebook_structure(self):
        """Test that the notebook has the expected structure."""
        # The notebook should have cells
        self.assertGreater(len(self.notebook['cells']), 10, "Notebook should have multiple cells")
        
        # First cell should be markdown with the title
        self.assertEqual(self.notebook['cells'][0]['cell_type'], 'markdown', "First cell should be markdown")
        first_cell = self.cell_sources[0]
        self.assertIn("Neural Plasticity Demo", first_cell, "Title should be present")
        self.assertIn("v0.0.", first_cell, "Version number should be present")
        
        # Check for config cell
        config_found = False
        for source in self.cell_sources:
            if "MODEL_NAME = " in source and "DATASET = " in source and "ENABLE_LONG_TRAINING = " in source:
                config_found = True
                break
        self.assertTrue(config_found, "Config cell should be present")
        
        # Check for model loading
        model_loading_found = False
        for source in self.cell_sources:
            if "model = AutoModelForCausalLM.from_pretrained" in source:
                model_loading_found = True
                break
        self.assertTrue(model_loading_found, "Model loading cell should be present")
        
        # Check for visualize_gradient_norms function
        viz_function_found = False
        for source in self.cell_sources:
            if "def visualize_gradient_norms" in source:
                viz_function_found = True
                break
        self.assertTrue(viz_function_found, "visualize_gradient_norms function should be defined")
    
    def test_memory_management(self):
        """Test that the notebook has memory management for Colab stability."""
        memory_mgmt_found = False
        for source in self.cell_sources:
            if "def clear_memory" in source and "torch.cuda.empty_cache" in source:
                memory_mgmt_found = True
                break
        self.assertTrue(memory_mgmt_found, "Memory management function should be present")
        
        # ENABLE_LONG_TRAINING should be False for stability
        for source in self.cell_sources:
            if "ENABLE_LONG_TRAINING = " in source:
                self.assertIn("ENABLE_LONG_TRAINING = False", source, "ENABLE_LONG_TRAINING should be False for stability")
    
    def test_error_handling(self):
        """Test that the notebook has proper error handling."""
        # Check for checkpoint saving on error
        error_handling_found = False
        for source in self.cell_sources:
            if "except Exception as e:" in source and "save_checkpoint" in source:
                error_handling_found = True
                break
        self.assertTrue(error_handling_found, "Error handling with checkpoint saving should be present")
        
        # Check for memory error handling
        memory_error_handling_found = False
        for source in self.cell_sources:
            if "except (MemoryError, RuntimeError)" in source:
                memory_error_handling_found = True
                break
        self.assertTrue(memory_error_handling_found, "Memory error handling should be present")

    def test_proper_checkpointing(self):
        """Test that the notebook has proper checkpointing configuration."""
        checkpoint_interval_found = False
        for source in self.cell_sources:
            if "CHECKPOINT_INTERVAL = " in source:
                self.assertIn("CHECKPOINT_INTERVAL = 500", source, "Checkpoint interval should be 500 for stability")
                checkpoint_interval_found = True
                break
        self.assertTrue(checkpoint_interval_found, "CHECKPOINT_INTERVAL should be configured")

    def test_visualization_functions(self):
        """Test that the visualization functions have proper scaling."""
        # At least one cell should have proper attention visualization with clim
        proper_attn_viz_found = False
        proper_entropy_viz_found = False
        
        for source in self.cell_sources:
            # Check attention visualization
            if "plt.imshow" in source and "attention" in source.lower():
                # Check if clim is used after imshow
                imshow_pos = source.find("plt.imshow")
                clim_pos = source.find("plt.clim", imshow_pos)
                if clim_pos > imshow_pos:
                    proper_attn_viz_found = True
            
            # Check for entropy visualization with specified range
            if "entropy" in source.lower() and "plt.imshow" in source and "plt.clim" in source:
                proper_entropy_viz_found = True
        
        self.assertTrue(proper_attn_viz_found, "At least one cell should have proper attention visualization with clim")
        self.assertTrue(proper_entropy_viz_found, "At least one cell should have proper entropy visualization with clim")


if __name__ == "__main__":
    unittest.main()