#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity Experiment Runner

This script executes neural plasticity experiments using the standardized
object-oriented architecture and outputs results to the /output directory.
It leverages the comprehensive NeuralPlasticityExperiment class to provide
a unified workflow for neural plasticity research.

Version: v0.0.40 (2025-04-20 21:30:00)
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime

# Add project root to path if needed
if __name__ == "__main__":
    # Configure paths and environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import project configuration
    try:
        from config.paths import OUTPUT_DIR
    except ImportError:
        # Fallback if config module is not available
        OUTPUT_DIR = os.path.join(project_root, "output")

    # Configure logging with both console and file output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_dir = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(logging_dir, exist_ok=True)
    
    log_file = os.path.join(logging_dir, f"neural_plasticity_experiment_{timestamp}.log")
    
    # Configure logging format with timestamps
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, 
                       format=log_format,
                       handlers=[
                           logging.FileHandler(log_file),
                           logging.StreamHandler()
                       ])
    
    logger = logging.getLogger("neural_plasticity")
    logger.info(f"Starting neural plasticity experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    
    # Setup output directory within the project's output directory
    output_dir = os.path.join(OUTPUT_DIR, f"neural_plasticity_{timestamp}")
    dashboard_dir = os.path.join(output_dir, "dashboards")
    models_dir = os.path.join(output_dir, "models")
    visualizations_dir = os.path.join(output_dir, "visualizations")
    
    # Create all necessary directories
    for directory in [output_dir, dashboard_dir, models_dir, visualizations_dir]:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Created directory: {directory}")
    
    # Parse command line arguments with comprehensive options
    import argparse
    parser = argparse.ArgumentParser(
        description="Run neural plasticity experiment with comprehensive analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration group
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_name", type=str, default="distilgpt2", 
                           help="Model name or path (e.g., distilgpt2, gpt2, facebook/opt-125m)")
    model_group.add_argument("--device", type=str, default=None, 
                           help="Device to run on (cpu, cuda, auto). Auto-detected if not specified.")
    
    # Dataset configuration group
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument("--dataset", type=str, default="wikitext",
                          help="Dataset name (e.g., wikitext, cnn_dailymail)")
    data_group.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                          help="Dataset configuration")
    data_group.add_argument("--batch_size", type=int, default=None,
                          help="Batch size (default: 4 for quick test, 8 for full run)")
    data_group.add_argument("--max_length", type=int, default=128,
                          help="Maximum sequence length")
    
    # Pruning configuration group
    pruning_group = parser.add_argument_group("Pruning Configuration")
    pruning_group.add_argument("--pruning_strategy", type=str, default="entropy", 
                             choices=["entropy", "magnitude", "random", "combined"],
                             help="Pruning strategy to use")
    pruning_group.add_argument("--pruning_level", type=float, default=0.2, 
                             help="Pruning level (0.0 to 1.0)")
    pruning_group.add_argument("--learning_rate", type=float, default=5e-5,
                             help="Learning rate for training")
    pruning_group.add_argument("--cycles", type=int, default=1, 
                             help="Number of pruning cycles to run")
    pruning_group.add_argument("--training_steps", type=int, default=100, 
                             help="Number of training steps per cycle")
    
    # Experiment mode group
    mode_group = parser.add_argument_group("Experiment Mode")
    mode_group.add_argument("--quick_test", action="store_true", 
                          help="Run quick test with minimal data and faster execution")
    mode_group.add_argument("--compare_strategies", action="store_true",
                          help="Compare multiple pruning strategies and levels")
    
    # Output configuration group
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--output_dir", type=str, default=output_dir, 
                            help="Output directory for experiment results")
    output_group.add_argument("--save_model", action="store_true", 
                            help="Save the model after experiment")
    output_group.add_argument("--no_visualize", action="store_true", 
                            help="Disable visualization generation")
    output_group.add_argument("--use_dashboard", action="store_true", 
                            help="Generate interactive HTML dashboard")
    output_group.add_argument("--verbose", action="store_true", default=True,
                            help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Log key experiment settings
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset}/{args.dataset_config}")
    logger.info(f"Pruning Strategy: {args.pruning_strategy}")
    logger.info(f"Pruning Level: {args.pruning_level}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    # Ensure all output paths are absolute
    args.output_dir = os.path.abspath(args.output_dir)
    dashboard_path = os.path.join(args.output_dir, "dashboard.html")
    
    # Set batch size based on mode if not explicitly specified
    if args.batch_size is None:
        args.batch_size = 4 if args.quick_test else 8
    
    # Import the experiment module
    try:
        from utils.neural_plasticity.experiment import NeuralPlasticityExperiment
        from utils.neural_plasticity.dashboard.reporter import DashboardReporter
        logger.info("Successfully imported NeuralPlasticityExperiment and DashboardReporter")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure the neural_plasticity module is installed correctly")
        sys.exit(1)
    
    # Record experiment start time for performance measurement
    import time
    start_time = time.time()
    
    # Initialize dashboard for real-time monitoring
    wandb_dashboard = None
    if args.use_dashboard:
        try:
            logger.info("Initializing dashboard...")
            
            # Choose between regular and multi-phase dashboard
            if args.cycles > 1 or not args.quick_test:
                # Use multi-phase dashboard for complex experiments
                from utils.neural_plasticity.dashboard.multi_phase_dashboard import MultiPhaseDashboard
                dashboard_class = MultiPhaseDashboard
                logger.info("Using multi-phase dashboard for comprehensive tracking")
            else:
                # Use basic dashboard for quick tests
                from utils.neural_plasticity.dashboard.wandb_integration import WandbDashboard
                dashboard_class = WandbDashboard
            
            # Create a timestamp-based experiment name
            experiment_name = f"np-{args.model_name.split('/')[-1]}-{args.pruning_strategy}-{timestamp}"
            
            # Initialize the dashboard
            wandb_dashboard = dashboard_class(
                project_name="neural-plasticity",
                experiment_name=experiment_name,
                output_dir=os.path.join(OUTPUT_DIR, "wandb"),
                config={
                    "model_name": args.model_name,
                    "dataset": f"{args.dataset}/{args.dataset_config}",
                    "pruning_strategy": args.pruning_strategy,
                    "pruning_level": args.pruning_level,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "cycles": args.cycles,
                    "quick_test": args.quick_test,
                    "training_steps": args.training_steps
                },
                mode="online" if not args.quick_test else "offline",  # Use offline mode for quick tests
                tags=["sentinel-ai", "multi-phase" if args.cycles > 1 else "single-cycle"]
            )
            
            # Set initial phase
            wandb_dashboard.set_phase("setup")
            wandb_dashboard.record_phase_transition("setup", 0)
            logger.info(f"Dashboard initialized: {dashboard_class.__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            logger.warning("Continuing without real-time dashboard")
            args.use_dashboard = False
    
    # Run experiment
    try:
        logger.info("Creating experiment instance...")
        
        # Create metrics callback if wandb dashboard is enabled
        metrics_callback = wandb_dashboard.get_metrics_callback() if wandb_dashboard else None
        sample_callback = wandb_dashboard.get_sample_callback() if wandb_dashboard else None
        
        if wandb_dashboard:
            # Log that we're starting experiment creation
            wandb_dashboard.log_metrics({"status": "running", "message": "Creating experiment instance"})
        
        # Create experiment with comprehensive configuration
        experiment = NeuralPlasticityExperiment(
            # Model parameters
            model_name=args.model_name,
            device=args.device,
            
            # Dataset parameters
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            batch_size=args.batch_size,
            max_length=args.max_length,
            
            # Pruning parameters
            pruning_strategy=args.pruning_strategy,
            pruning_level=args.pruning_level,
            learning_rate=args.learning_rate,
            
            # Output parameters
            output_dir=args.output_dir,
            save_results=not args.no_visualize,
            use_dashboard=args.use_dashboard,
            dashboard_dir=dashboard_dir,
            
            # Behavior parameters
            verbose=args.verbose,
            show_samples=not args.quick_test,
            
            # Callbacks
            metrics_callback=metrics_callback,
            sample_callback=sample_callback
        )
        
        logger.info("Experiment instance created successfully")
        
        # Run the experiment differently based on selected mode
        if args.compare_strategies:
            # Compare different pruning strategies
            logger.info("Running strategy comparison experiment")
            if wandb_dashboard:
                wandb_dashboard.log_metrics({"status": "running", "message": "Comparing pruning strategies"})
                wandb_dashboard.set_phase("analysis")
            
            strategies = ["entropy", "magnitude", "combined", "random"]
            pruning_levels = [0.1, 0.2, 0.3]
            
            comparison_results = experiment.compare_pruning_strategies(
                strategies=strategies,
                pruning_levels=pruning_levels,
                cycles=1,
                training_steps=args.training_steps // 2 if args.quick_test else args.training_steps,
                save_path=os.path.join(visualizations_dir, "strategy_comparison.png")
            )
            
            logger.info("Strategy comparison completed")
            if wandb_dashboard:
                wandb_dashboard.log_metrics({"status": "completed", "message": "Strategy comparison completed"})
                wandb_dashboard.set_phase("complete")
        
        else:
            # Use the comprehensive run_full_experiment method for standard execution
            logger.info("Running full neural plasticity experiment")
            
            # Adjust parameters for quick test mode if enabled
            warmup_epochs = 1
            training_steps = min(25, args.training_steps // 10) if args.quick_test else args.training_steps
            
            if wandb_dashboard:
                wandb_dashboard.log_metrics({"status": "running", "message": "Starting warmup phase"})
                wandb_dashboard.set_phase("warmup")
                
                # Record phase transition for multi-phase dashboard
                if hasattr(wandb_dashboard, "record_phase_transition"):
                    current_step = getattr(wandb_dashboard, "current_step", 0)
                    wandb_dashboard.record_phase_transition("warmup", current_step)
            
            # Create a callback to track metrics and phases during execution
            def experiment_progress_callback(phase, step, metrics):
                """Callback to track experiment progress through phases."""
                if wandb_dashboard and hasattr(wandb_dashboard, "record_step"):
                    # Add phase info to metrics
                    metrics["phase"] = phase
                    # Record step with multi-phase dashboard
                    wandb_dashboard.record_step(metrics, step)
                
                # Regular wandb logging
                if wandb_dashboard:
                    wandb_dashboard.log_metrics(metrics, step)
            
            # Create a callback for tracking pruning events
            def pruning_event_callback(pruning_info, step):
                """Callback to track pruning events."""
                if wandb_dashboard and hasattr(wandb_dashboard, "record_pruning_event"):
                    wandb_dashboard.record_pruning_event(pruning_info, step)
                
                # Regular wandb logging
                if wandb_dashboard:
                    wandb_dashboard.log_pruning_decision(pruning_info, step)
            
            # Add these callbacks to the experiment options
            experiment_opts = {
                "warmup_epochs": warmup_epochs,
                "pruning_cycles": args.cycles,
                "training_steps": training_steps,
                "progress_callback": experiment_progress_callback,
                "pruning_callback": pruning_event_callback,
            }
            
            # Run the complete experiment pipeline
            results = experiment.run_full_experiment(**experiment_opts)
            
            # Extract key metrics for logging
            baseline_perplexity = results["baseline_metrics"]["perplexity"]
            final_perplexity = results["final_metrics"]["perplexity"]
            improvement_percent = results["improvement_percent"]
            
            # Update dashboard with final metrics
            if wandb_dashboard:
                wandb_dashboard.log_metrics({
                    "baseline_perplexity": baseline_perplexity,
                    "final_perplexity": final_perplexity,
                    "improvement_percent": improvement_percent
                })
                
                # Update pruning information if available
                if "pruned_heads" in results:
                    pruning_decision = {
                        "strategy": args.pruning_strategy,
                        "pruning_level": args.pruning_level,
                        "pruned_heads": results["pruned_heads"],
                        "cycle": args.cycles
                    }
                    wandb_dashboard.log_pruning_decision(pruning_decision)
                
                # Add the inference phase to show model evaluation
                if not args.quick_test:
                    wandb_dashboard.set_phase("evaluation")
                    logger.info("Running inference phase with comprehensive evaluation...")
                    
                    # Generate sample texts with both baseline and pruned models
                    prompts = [
                        "The future of artificial intelligence seems to be",
                        "Neural networks have revolutionized how we approach",
                        "The key challenge in machine learning today is"
                    ]
                    
                    # Generate samples from baseline and pruned models
                    generation_samples = {}
                    
                    # Get baseline model samples
                    for i, prompt in enumerate(prompts):
                        baseline_text = experiment.generate_baseline_text(prompt, max_length=100)
                        pruned_text = experiment.generate_text(prompt, max_length=100)
                        
                        # Store samples for dashboard
                        sample_id = f"sample_{i+1}"
                        generation_samples[f"baseline"] = {
                            "prompt": prompt,
                            "output": baseline_text
                        }
                        generation_samples[f"pruned"] = {
                            "prompt": prompt,
                            "output": pruned_text
                        }
                        
                        # Log individual samples for different visualizations
                        wandb_dashboard.log_text_sample(prompt, baseline_text, model_type="baseline")
                        wandb_dashboard.log_text_sample(prompt, pruned_text, model_type="pruned")
                        
                        # Create comparison visualization
                        wandb_dashboard.log_inference_comparison(
                            prompt, baseline_text, pruned_text,
                            metrics={
                                "baseline_perplexity": baseline_perplexity,
                                "pruned_perplexity": final_perplexity,
                                "improvement_percent": improvement_percent
                            }
                        )
                        
                        # Only need one detailed comparison for the dashboard
                        if i == 0:
                            break
                    
                    # Get attention visualization data if available
                    attention_data = experiment.get_attention_maps(prompts[0])
                    
                    # Create comprehensive inference dashboard
                    perplexity_data = {
                        "baseline": baseline_perplexity,
                        "pruned": final_perplexity,
                        "model_size_reduction": results.get("sparsity_percent", 0.0) * 100,
                        "inference_speedup": results.get("speedup", 1.0)
                    }
                    
                    # Log comprehensive inference dashboard
                    wandb_dashboard.log_inference_dashboard(
                        perplexity_data=perplexity_data,
                        generation_samples=generation_samples,
                        attention_data=attention_data
                    )
                    
                    logger.info("Inference evaluation dashboard completed")
                
                wandb_dashboard.log_metrics({"status": "completed", "message": "Experiment completed successfully"})
                wandb_dashboard.set_phase("complete")
            
            # Save the model if requested
            if args.save_model:
                logger.info("Saving trained model...")
                model_path = os.path.join(models_dir, "pruned_model")
                experiment.save_model(path=model_path)
                logger.info(f"Model saved to {model_path}")
            
            # Create metrics dashboards
            if not args.no_visualize:
                # Create standard metrics dashboard 
                dashboard_path = os.path.join(visualizations_dir, "metrics_dashboard.png")
                experiment.visualize_metrics_dashboard(save_path=dashboard_path)
                logger.info(f"Metrics dashboard saved to {dashboard_path}")
                
                # If using multi-phase dashboard, generate the comprehensive visualization
                if wandb_dashboard and hasattr(wandb_dashboard, "visualize_complete_process"):
                    # Path for multi-phase dashboard
                    multi_phase_path = os.path.join(visualizations_dir, "multi_phase_dashboard.png")
                    logger.info("Generating multi-phase dashboard visualization...")
                    wandb_dashboard.visualize_complete_process(save_path=multi_phase_path)
                    
                    # Create standalone HTML dashboard with all visualizations
                    html_path = os.path.join(dashboard_dir, "dashboard.html")
                    logger.info("Generating standalone HTML dashboard...")
                    html_result = wandb_dashboard.generate_standalone_dashboard(dashboard_dir)
                    logger.info(f"Standalone dashboard saved to: {html_result}")
                    
                    # Try to open the dashboard in a browser
                    try:
                        import webbrowser
                        logger.info(f"Opening dashboard in browser: file://{os.path.abspath(html_result)}")
                        webbrowser.open(f"file://{os.path.abspath(html_result)}")
                    except Exception as e:
                        logger.warning(f"Could not open browser: {e}")
        
        # Calculate and log execution time
        execution_time = time.time() - start_time
        minutes, seconds = divmod(execution_time, 60)
        
        # Log final results
        logger.info(f"Experiment completed successfully in {int(minutes)} minutes {int(seconds)} seconds")
        logger.info(f"Results saved to {args.output_dir}")
        
        if not args.compare_strategies:
            logger.info(f"Performance Summary:")
            logger.info(f"  Baseline perplexity: {baseline_perplexity:.2f}")
            logger.info(f"  Final perplexity: {final_perplexity:.2f}")
            logger.info(f"  Improvement: {improvement_percent:.2f}%")
            
            # Print summary with hyperparameters
            logger.info(f"Experiment Hyperparameters:")
            logger.info(f"  Model: {args.model_name}")
            logger.info(f"  Pruning Strategy: {args.pruning_strategy}")
            logger.info(f"  Pruning Level: {args.pruning_level}")
            logger.info(f"  Learning Rate: {args.learning_rate}")
            logger.info(f"  Batch Size: {args.batch_size}")
            logger.info(f"  Training Steps: {training_steps}")
            logger.info(f"  Pruning Cycles: {args.cycles}")
        
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running experiment: {e}", exc_info=True)
        
        # Try to save partial results if possible
        try:
            if 'experiment' in locals() and hasattr(experiment, 'save_metadata'):
                experiment.save_metadata(os.path.join(args.output_dir, "partial_results_metadata.json"))
                logger.info("Saved partial results metadata before exiting")
        except Exception as save_error:
            logger.error(f"Failed to save partial results: {save_error}")
        
        sys.exit(1)
    finally:
        # Clean up custom dashboard resources if started
        if 'dashboard_reporter' in locals() and dashboard_reporter:
            try:
                logger.info("Cleaning up custom dashboard resources...")
                dashboard_reporter.close()
            except Exception as e:
                logger.error(f"Error closing custom dashboard: {e}")
        
        # Clean up wandb dashboard if started
        if 'wandb_dashboard' in locals() and wandb_dashboard:
            try:
                logger.info("Cleaning up Weights & Biases dashboard resources...")
                wandb_dashboard.finish()
            except Exception as e:
                logger.error(f"Error finishing wandb dashboard: {e}")
        
        # Final cleanup
        logger.info(f"Experiment session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")