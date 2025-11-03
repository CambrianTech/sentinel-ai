"""
Integration test for the PlasticityController.

This test verifies that the PlasticityController works correctly with real models,
demonstrating how to integrate it into a training loop with appropriate metrics tracking.

To run:
    pytest -xvs tests/sentinel/pruning/integration_test_plasticity.py
"""

import torch
import pytest
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset

from sentinel.pruning.plasticity_controller import create_plasticity_controller, PlasticityDecision
from sentinel.pruning.dual_mode_pruning import PruningMode


@pytest.mark.integration
class TestPlasticityControllerIntegration:
    """Integration tests for the plasticity controller."""

    @pytest.fixture(scope="class")
    def model_and_dataloader(self):
        """Set up model and dataloader for testing."""
        # Load a lightweight model for testing
        model_name = "distilgpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load a small dataset sample
        try:
            # Use a tiny subset of wikitext for testing
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:20]")
            
            # Tokenize the dataset
            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)
                
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["text"])
            tokenized_dataset = tokenized_dataset.with_format("torch")
            
            # Create a small dataloader
            dataloader = DataLoader(
                tokenized_dataset,
                batch_size=2,
                collate_fn=default_data_collator
            )
        except Exception as e:
            pytest.skip(f"Failed to load dataset: {e}")
        
        return model, dataloader, tokenizer
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
    def test_plasticity_step_with_gpu(self, model_and_dataloader, tmp_path):
        """Test a complete plasticity controller step with GPU."""
        model, dataloader, _ = model_and_dataloader
        model = model.to("cuda")
        
        # Create controller
        controller = create_plasticity_controller(
            model=model,
            mode=PruningMode.ADAPTIVE,
            high_entropy_threshold=0.6,  # Lower for testing
            low_entropy_threshold=0.3,
            grad_threshold=1e-4  # Lower for testing
        )
        
        # Run a plasticity step
        output_dir = os.path.join(tmp_path, "plasticity_test_gpu")
        pruned, revived, metrics = controller.step(
            dataloader=dataloader,
            num_batches=1,
            verbose=True,
            output_dir=output_dir
        )
        
        # Basic assertions
        assert isinstance(pruned, list), "Should return a list of pruned heads"
        assert isinstance(revived, list), "Should return a list of revived heads"
        assert "entropy" in metrics, "Should include entropy in metrics"
        assert "grad_norm" in metrics, "Should include grad_norm in metrics"
        
        # There should be output files
        assert os.path.exists(output_dir), "Output directory should be created"
        
        # Return model to CPU to free GPU memory
        model = model.to("cpu")
    
    def test_plasticity_step_with_cpu(self, model_and_dataloader, tmp_path):
        """Test a complete plasticity controller step with CPU."""
        model, dataloader, _ = model_and_dataloader
        
        # Create controller
        controller = create_plasticity_controller(
            model=model,
            mode=PruningMode.ADAPTIVE,
            high_entropy_threshold=0.6,  # Lower for testing
            low_entropy_threshold=0.3,
            grad_threshold=1e-4  # Lower for testing 
        )
        
        # Run a plasticity step
        output_dir = os.path.join(tmp_path, "plasticity_test_cpu")
        pruned, revived, metrics = controller.step(
            dataloader=dataloader,
            num_batches=1,
            verbose=True,
            output_dir=output_dir
        )
        
        # Basic assertions
        assert isinstance(pruned, list), "Should return a list of pruned heads"
        assert isinstance(revived, list), "Should return a list of revived heads"
        assert "entropy" in metrics, "Should include entropy in metrics"
        assert "grad_norm" in metrics, "Should include grad_norm in metrics"
        assert "sparsity" in metrics, "Should include sparsity in metrics"
        
        # There should be output files
        assert os.path.exists(output_dir), "Output directory should be created"
    
    def test_multi_cycle_pruning(self, model_and_dataloader, tmp_path):
        """Test multiple cycles of plasticity to ensure the pruning state is maintained."""
        model, dataloader, _ = model_and_dataloader
        
        # Create controller
        controller = create_plasticity_controller(
            model=model,
            mode=PruningMode.ADAPTIVE,
            high_entropy_threshold=0.6,  # Lower for testing
            low_entropy_threshold=0.3,
            grad_threshold=1e-4  # Lower for testing
        )
        
        output_dir = os.path.join(tmp_path, "multi_cycle_test")
        
        # Run multiple plasticity steps
        metrics_history = []
        for i in range(3):  # Just a few steps for testing
            pruned, revived, metrics = controller.step(
                dataloader=dataloader,
                num_batches=1,
                verbose=False,
                output_dir=os.path.join(output_dir, f"step_{i}")
            )
            metrics_history.append({
                "pruned": len(pruned),
                "revived": len(revived),
                "total_pruned": metrics["total_pruned"],
                "sparsity": metrics["sparsity"]
            })
        
        # Get a summary of the controller state
        summary = controller.get_summary()
        
        # Check that the model is still in a valid state
        assert summary["total_heads"] > 0, "Should have valid head count"
        assert summary["pruned_heads"] >= 0, "Should have valid pruned head count"
        assert 0 <= summary["pruning_rate"] <= 1, "Pruning rate should be between 0 and 1"
        assert "sparsity" in summary, "Should include sparsity in summary"
        
        # Check that metrics are being tracked correctly across cycles
        assert len(metrics_history) == 3, "Should have metrics for all cycles"
    
    def test_adaptive_mode_recovery(self, model_and_dataloader):
        """Test that the controller can recover heads in adaptive mode."""
        model, dataloader, _ = model_and_dataloader
        
        # Create controller with VERY aggressive pruning for testing
        controller = create_plasticity_controller(
            model=model,
            mode=PruningMode.ADAPTIVE,  # Adaptive mode allows revival
            high_entropy_threshold=0.3,  # Extremely low threshold
            low_entropy_threshold=0.1,
            grad_threshold=1e-5,  # Very low threshold
            min_zero_epochs=1  # Quick revival for testing
        )
        
        # First step should prune some heads
        pruned1, _, _ = controller.step(dataloader, num_batches=1)
        
        # Check if any heads were pruned
        if not pruned1:
            pytest.skip("No heads were pruned, cannot test revival")
        
        # Manually set some heads as candidates for revival
        for layer_idx, head_idx in pruned1[:1]:  # Just use the first pruned head
            # Set stats to encourage revival
            s = controller.stats[layer_idx][head_idx]
            s['entropy'] = [0.05]  # Very low entropy (focused)
            s['grad_norm'] = [0.1]  # Very high gradient (important)
            s['zeroed_epochs'] = 1  # Minimum epochs required
        
        # Second step should revive some heads
        _, revived, _ = controller.step(dataloader, num_batches=1)
        
        # In adaptive mode, some heads should be revived (we explicitly set them up to be)
        assert len(revived) > 0, "Should revive some heads in adaptive mode"
    
    def test_compressed_mode_no_recovery(self, model_and_dataloader):
        """Test that the controller does not recover heads in compressed mode."""
        model, dataloader, _ = model_and_dataloader
        
        # Create controller with aggressive pruning in COMPRESSED mode
        controller = create_plasticity_controller(
            model=model,
            mode=PruningMode.COMPRESSED,  # Compressed mode prevents revival
            high_entropy_threshold=0.3,
            low_entropy_threshold=0.1,
            grad_threshold=1e-5,
            min_zero_epochs=1
        )
        
        # First step should prune some heads
        pruned1, _, _ = controller.step(dataloader, num_batches=1)
        
        # Check if any heads were pruned
        if not pruned1:
            pytest.skip("No heads were pruned, cannot test revival prevention")
        
        # Manually set some heads as candidates for revival (same as above)
        for layer_idx, head_idx in pruned1[:1]:
            s = controller.stats[layer_idx][head_idx]
            s['entropy'] = [0.05]
            s['grad_norm'] = [0.1]
            s['zeroed_epochs'] = 1
        
        # Second step should NOT revive any heads in COMPRESSED mode
        _, revived, _ = controller.step(dataloader, num_batches=1)
        
        # In compressed mode, no heads should be revived
        assert len(revived) == 0, "Should NOT revive heads in compressed mode"