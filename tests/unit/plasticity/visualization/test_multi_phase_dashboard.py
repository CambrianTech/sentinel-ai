#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit Tests for Multi-Phase Dashboard with Controller Integration

Tests the functionality of the MultiPhaseDashboard class for visualizing
neural plasticity experiments with ANN controller integration.

Version: v0.1.0 (2025-04-20 20:05:00)
"""

import os
import sys
import pytest
import numpy as np
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import json
import shutil

# Add project root to path if needed for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sentinel.plasticity.visualization.multi_phase.dashboard import MultiPhaseDashboard


class TestMultiPhaseDashboard:
    """Tests for the MultiPhaseDashboard implementation."""

    @pytest.fixture
    def dashboard(self):
        """Create a temporary dashboard for testing."""
        temp_dir = tempfile.mkdtemp()
        dashboard = MultiPhaseDashboard(
            output_dir=temp_dir,
            experiment_name="test_experiment",
            config={"test": True}
        )
        yield dashboard
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def dashboard_with_data(self, dashboard):
        """Create a dashboard with some test data."""
        # Add phase transitions
        dashboard.record_phase_transition("warmup", 0)
        dashboard.record_phase_transition("analysis", 50)
        dashboard.record_phase_transition("pruning", 60)
        dashboard.record_phase_transition("fine-tuning", 61)
        
        # Add metrics
        for i in range(100):
            metrics = {
                "loss": 5.0 - i * 0.03 + np.random.normal(0, 0.1),
                "perplexity": np.exp((5.0 - i * 0.03) * 0.5),
                "sparsity": 0.0 if i < 60 else 0.2
            }
            dashboard.record_step(metrics, i)
        
        # Add pruning event
        pruning_info = {
            "strategy": "entropy",
            "pruning_level": 0.2,
            "pruned_heads": ["0.1", "1.3", "2.4"],
            "cycle": 1
        }
        dashboard.record_pruning_event(pruning_info, 60)
        
        # Add head metrics
        head_metrics = {
            "0.0": {"entropy": 1.2, "magnitude": 0.8},
            "0.1": {"entropy": 0.5, "magnitude": 0.3},
            "1.0": {"entropy": 1.4, "magnitude": 0.9}
        }
        dashboard.record_head_metrics(head_metrics, 50)
        
        # Add controller decision
        decision = {
            "active_heads": ["0.0", "1.0", "1.1"],
            "controller_loss": 0.4
        }
        dashboard.record_controller_decision(decision, 70)
        
        return dashboard

    def test_initialization(self, dashboard):
        """Test that the dashboard initializes correctly."""
        assert dashboard.experiment_name == "test_experiment"
        assert dashboard.config == {"test": True}
        assert dashboard.phases == []
        assert dashboard.phase_transitions == []
        assert dashboard.current_phase is None

    def test_record_phase_transition(self, dashboard):
        """Test recording phase transitions."""
        dashboard.record_phase_transition("warmup", 0)
        dashboard.record_phase_transition("pruning", 50)
        
        assert dashboard.phases == ["warmup", "pruning"]
        assert dashboard.phase_transitions == [0, 50]
        assert dashboard.current_phase == "pruning"

    def test_record_step(self, dashboard):
        """Test recording metrics for a step."""
        dashboard.record_step({"loss": 5.0, "perplexity": 20.0}, 0)
        dashboard.record_step({"loss": 4.5, "perplexity": 18.0}, 1)
        
        assert len(dashboard.all_steps) == 2
        assert dashboard.all_steps == [0, 1]
        assert dashboard.all_loss == [5.0, 4.5]
        assert dashboard.all_perplexity == [20.0, 18.0]

    def test_record_pruning_event(self, dashboard):
        """Test recording pruning events."""
        pruning_info = {
            "strategy": "entropy", 
            "pruning_level": 0.2,
            "pruned_heads": ["0.1", "1.3", "2.4"],
            "cycle": 1
        }
        dashboard.record_pruning_event(pruning_info, 50)
        
        assert len(dashboard.pruning_info) == 1
        assert dashboard.pruning_info[0]["strategy"] == "entropy"
        assert dashboard.pruning_info[0]["pruned_heads"] == ["0.1", "1.3", "2.4"]
        assert dashboard.pruned_heads == {"0.1", "1.3", "2.4"}

    def test_record_head_metrics(self, dashboard):
        """Test recording head metrics."""
        head_metrics = {
            "0.0": {"entropy": 1.2, "magnitude": 0.8},
            "0.1": {"entropy": 0.5, "magnitude": 0.3}
        }
        dashboard.record_head_metrics(head_metrics, 50)
        
        assert "0.0" in dashboard.head_metrics
        assert "0.1" in dashboard.head_metrics
        assert dashboard.head_metrics["0.0"]["steps"] == [50]
        assert "entropy" in dashboard.head_metrics["0.0"]["metrics"]
        assert dashboard.head_metrics["0.0"]["metrics"]["entropy"] == [1.2]

    def test_record_controller_decision(self, dashboard):
        """Test recording controller decisions."""
        decision = {
            "active_heads": ["0.0", "1.0", "1.1"],
            "controller_loss": 0.4
        }
        dashboard.record_controller_decision(decision, 70)
        
        assert len(dashboard.controller_decisions) == 1
        assert dashboard.controller_decisions[0]["step"] == 70
        assert dashboard.controller_decisions[0]["active_heads"] == ["0.0", "1.0", "1.1"]
        assert dashboard.active_heads == {"0.0", "1.0", "1.1"}

    def test_visualize_complete_process(self, dashboard_with_data):
        """Test generating the complete process visualization."""
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            fig = dashboard_with_data.visualize_complete_process(tmp.name)
            assert fig is not None
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0

    def test_generate_controller_dashboard(self, dashboard_with_data):
        """Test generating the controller dashboard."""
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            fig = dashboard_with_data.generate_controller_dashboard(tmp.name)
            assert fig is not None
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0

    def test_generate_standalone_dashboard(self, dashboard_with_data):
        """Test generating the standalone HTML dashboard."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = dashboard_with_data.generate_standalone_dashboard(tmp_dir)
            assert os.path.exists(html_path)
            assert os.path.getsize(html_path) > 0
            
            # Check that required images were generated
            assert os.path.exists(os.path.join(tmp_dir, "complete_process.png"))

    def test_save_dashboard_data(self, dashboard_with_data):
        """Test saving dashboard data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = dashboard_with_data.save_dashboard_data(tmp_dir)
            assert os.path.exists(data_path)
            
            # Check that data files were created
            assert os.path.exists(os.path.join(data_path, "dashboard_data.json"))
            assert os.path.exists(os.path.join(data_path, "head_scores.json"))
            
            # Verify JSON is valid
            with open(os.path.join(data_path, "dashboard_data.json"), 'r') as f:
                data = json.load(f)
                assert "experiment_metadata" in data
                assert "metrics" in data
                assert "phases" in data

    def test_compatibility_methods(self, dashboard):
        """Test WandbDashboard compatibility methods."""
        # Test set_phase
        dashboard.set_phase("warmup")
        assert dashboard.current_phase == "warmup"
        
        # Test log_metrics
        dashboard.log_metrics({"loss": 5.0}, 0)
        assert dashboard.all_loss == [5.0]
        
        # Test log_pruning_decision
        pruning_info = {"pruned_heads": ["0.1"]}
        dashboard.log_pruning_decision(pruning_info)
        assert len(dashboard.pruning_info) == 1
        
        # Test callbacks
        metrics_callback = dashboard.get_metrics_callback()
        metrics_callback({"loss": 4.5}, 1)
        assert dashboard.all_loss == [5.0, 4.5]
        
        sample_callback = dashboard.get_sample_callback()
        sample_callback("Generated text", "Prompt", "pruned")
        assert hasattr(dashboard, "text_samples")
        assert len(dashboard.text_samples) == 1