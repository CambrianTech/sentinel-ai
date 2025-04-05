#!/usr/bin/env python
"""
Neural Plasticity Demo

This script demonstrates the complete neural plasticity tracking system,
showing how entropy journal, function tracking, stress protocols, and
visualization work together to study model adaptation.

The demo:
1. Initializes a small transformer model
2. Tracks entropy and function through plasticity cycles
3. Applies stress protocols to test adaptation
4. Creates visualizations of the entire process

This is a scientific instrument for studying the plasticity of transformer models.
"""

import os
import sys
import logging
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

# Import transformer model components
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Import our neural plasticity modules
from sentinel.plasticity.entropy_journal import EntropyJournal, EntropyJournalConfig
from sentinel.plasticity.function_tracking import FunctionTracker, FunctionTrackingConfig
from sentinel.plasticity.stress_protocols import (
    TaskAlternationProtocol, StressProtocolConfig
)
from sentinel.visualization.entropy_rhythm_plot import (
    plot_entropy_rhythm, create_animated_entropy_rhythm, create_entropy_delta_heatmap
)


def create_simple_dataset(tokenizer, texts, output_dir, block_size=128):
    """Create a simple dataset for training and evaluation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Write texts to a file
    dataset_path = os.path.join(output_dir, "text.txt")
    with open(dataset_path, "w") as f:
        f.write("\n".join(texts))
    
    # Create dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=block_size
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    return dataset, data_collator


def fine_tune_model(model, dataset, data_collator, output_dir, steps=100, batch_size=4):
    """Fine-tune the model for a small number of steps"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        save_steps=steps,
        save_total_limit=1,
        max_steps=steps,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    return model


def create_task_datasets(tokenizer, output_dir):
    """Create datasets for different tasks"""
    # Task 1: General text
    general_texts = [
        "The transformer architecture has revolutionized natural language processing by allowing models to process sequences in parallel.",
        "Self-attention mechanisms enable each position in a sequence to attend to all other positions, capturing dependencies across the entire sequence.",
        "The field of artificial intelligence combines computer science, mathematics, and cognitive science to create systems capable of learning and problem-solving.",
        "Transformer models have become the foundation for various language tasks, including translation, summarization, and text generation.",
        "Machine learning algorithms analyze data to identify patterns and make predictions without explicit programming rules.",
    ] * 10  # Repeat to get more samples
    
    # Task 2: Scientific text
    scientific_texts = [
        "Quantum mechanics describes the behavior of matter and energy at the smallest scales, revealing counterintuitive phenomena like wave-particle duality.",
        "Neural networks consist of interconnected layers of artificial neurons that transform input data through weighted connections and activation functions.",
        "Thermodynamics governs energy transfer and transformation, establishing principles like entropy increase in isolated systems over time.",
        "General relativity explains gravity as the curvature of spacetime caused by mass and energy, predicting phenomena like gravitational waves.",
        "Photosynthesis converts light energy into chemical energy, enabling plants to synthesize glucose from carbon dioxide and water.",
    ] * 10
    
    # Create datasets
    general_dataset, general_collator = create_simple_dataset(
        tokenizer, general_texts, os.path.join(output_dir, "general")
    )
    
    scientific_dataset, scientific_collator = create_simple_dataset(
        tokenizer, scientific_texts, os.path.join(output_dir, "scientific")
    )
    
    return {
        "general": (general_dataset, general_collator),
        "scientific": (scientific_dataset, scientific_collator)
    }


def create_data_loaders(tokenizer, task_datasets, batch_size=4):
    """Create data loaders from datasets"""
    from torch.utils.data import DataLoader
    
    def collate_fn(examples):
        # Convert examples to tensors
        batch = tokenizer.pad(
            examples,
            return_tensors="pt",
            padding="longest"
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
    
    data_loaders = {}
    for task_name, (dataset, _) in task_datasets.items():
        data_loaders[task_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn
        )
    
    return data_loaders


def run_neural_plasticity_demo(args):
    """Run the complete neural plasticity demo"""
    logger.info("Starting neural plasticity demo")
    
    # Create output directory
    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"plasticity_demo_{experiment_time}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create datasets for different tasks
    logger.info("Creating task datasets")
    task_datasets = create_task_datasets(tokenizer, output_dir)
    data_loaders = create_data_loaders(tokenizer, task_datasets, batch_size=args.batch_size)
    
    # Create test prompts for function tracking
    test_prompts = [
        "The transformer architecture allows",
        "Neural networks consist of",
        "Machine learning algorithms can",
        "Quantum mechanics describes",
        "The principles of thermodynamics state",
    ]
    
    # Initialize tracking systems
    logger.info("Initializing tracking systems")
    
    # Entropy journal
    entropy_config = EntropyJournalConfig(
        output_dir=os.path.join(output_dir, "entropy_journal"),
        experiment_name="plasticity_demo",
        create_visualizations=True
    )
    entropy_journal = EntropyJournal(entropy_config, device)
    
    # Function tracker
    function_config = FunctionTrackingConfig(
        output_dir=os.path.join(output_dir, "function_tracking"),
        experiment_name="plasticity_demo"
    )
    function_tracker = FunctionTracker(function_config, device)
    
    # Run plasticity cycles
    logger.info(f"Running {args.cycles} plasticity cycles")
    
    # Store model versions
    model_versions = {0: model.state_dict().copy()}
    
    # Record initial state
    logger.info("Recording initial model state")
    entropy_journal.record_model_state(
        model, 
        data_loaders["general"], 
        cycle_idx=0, 
        cycle_name="Initial"
    )
    
    # Run fine-tuning cycles, alternating between tasks
    for cycle in range(1, args.cycles + 1):
        logger.info(f"Running plasticity cycle {cycle}")
        
        # Choose task
        task = "general" if cycle % 2 == 1 else "scientific"
        logger.info(f"Fine-tuning on task: {task}")
        
        # Fine-tune the model
        dataset, data_collator = task_datasets[task]
        fine_tune_dir = os.path.join(output_dir, f"finetune_cycle_{cycle}")
        
        model = fine_tune_model(
            model, 
            dataset, 
            data_collator, 
            fine_tune_dir, 
            steps=args.steps_per_cycle,
            batch_size=args.batch_size
        )
        
        # Store this model version
        model_versions[cycle] = model.state_dict().copy()
        
        # Record entropy state
        logger.info(f"Recording model state after cycle {cycle}")
        entropy_journal.record_model_state(
            model, 
            data_loaders[task], 
            cycle_idx=cycle, 
            cycle_name=f"Cycle_{cycle}_{task}",
            metadata={"task": task}
        )
        
        # Track function preservation from previous cycle
        if cycle > 1:
            logger.info(f"Tracking function preservation between cycles {cycle-1} and {cycle}")
            
            # Load previous model version
            prev_model = AutoModelForCausalLM.from_pretrained(args.model_name)
            prev_model.load_state_dict(model_versions[cycle-1])
            prev_model = prev_model.to(device)
            
            # Track function
            function_tracker.track_function(
                prev_model,
                model,
                test_prompts,
                tokenizer,
                cycle_idx=cycle,
                cycle_name=f"Cycle_{cycle}_{task}"
            )
    
    # Create summary visualizations
    logger.info("Creating summary visualizations")
    
    # Entropy evolution
    entropy_journal.visualize_entropy_evolution()
    
    # Gate evolution (if model has gates)
    entropy_journal.visualize_gate_evolution()
    
    # Create entropy rhythm plots
    logger.info("Creating entropy rhythm plots")
    
    # Load data from journal
    journal_path = os.path.join(output_dir, "entropy_journal/plasticity_demo/entropy_journal.jsonl")
    
    # Create static EEG-like plot
    rhythm_plot_path = os.path.join(output_dir, "entropy_journal/plasticity_demo/visualizations/entropy_rhythm.png")
    plot_entropy_rhythm_from_file(
        journal_path,
        rhythm_plot_path,
        normalize=True,
        smooth_window=2,
        title="Entropy Rhythm Across Plasticity Cycles"
    )
    
    # Create animated plot
    animated_path = os.path.join(output_dir, "entropy_journal/plasticity_demo/visualizations/entropy_rhythm_animated.mp4")
    create_animated_entropy_rhythm_from_file(
        journal_path,
        animated_path,
        fps=5,
        normalize=True,
        title="Entropy Evolution Across Plasticity Cycles"
    )
    
    # Create delta heatmap
    delta_path = os.path.join(output_dir, "entropy_journal/plasticity_demo/visualizations/entropy_delta.png")
    create_entropy_delta_heatmap_from_file(
        journal_path,
        delta_path,
        title="Entropy Changes Between Cycles"
    )
    
    # Create function tracking summary
    logger.info("Creating function tracking summary")
    function_tracker.create_summary_report()
    
    # Run stress protocol if requested
    if args.run_stress_protocol:
        logger.info("Running stress protocol")
        
        # Create stress protocol config
        stress_config = StressProtocolConfig(
            output_dir=os.path.join(output_dir, "stress_protocol"),
            experiment_name="task_alternation_stress",
            protocol_type="task_alternation",
            cycles=args.stress_cycles,
            tasks=["general", "scientific"]
        )
        
        # Create fine-tuning function for the protocol
        def fine_tuning_fn(model, dataloader, steps=100):
            device = model.device
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            
            for _ in tqdm(range(steps), desc="Fine-tuning"):
                for batch in dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    break  # Just do one batch per step
        
        # Initialize protocol
        protocol = TaskAlternationProtocol(stress_config, device)
        
        # Run protocol
        protocol.run_protocol(
            model,
            data_loaders,
            fine_tuning_fn=fine_tuning_fn
        )
    
    logger.info(f"Demo completed successfully. Results saved to {output_dir}")


def create_entropy_delta_heatmap_from_file(journal_path, save_path, title=None):
    """Create entropy delta heatmap from a journal file"""
    import pandas as pd
    import json
    
    # Load journal
    entries = []
    with open(journal_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    
    df = pd.DataFrame(entries)
    
    # Create delta heatmap
    create_entropy_delta_heatmap(
        df,
        save_path,
        title=title
    )


def plot_entropy_rhythm_from_file(journal_path, save_path, normalize=True, smooth_window=1, title=None):
    """Create an entropy rhythm plot from a journal file"""
    import pandas as pd
    import json
    
    # Load journal
    entries = []
    with open(journal_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    
    df = pd.DataFrame(entries)
    
    # Create plot
    plot_entropy_rhythm(
        df,
        save_path=save_path,
        normalize=normalize,
        smooth_window=smooth_window,
        title=title
    )


def create_animated_entropy_rhythm_from_file(journal_path, save_path, fps=10, normalize=True, title=None):
    """Create an animated entropy rhythm plot from a journal file"""
    import pandas as pd
    import json
    
    # Load journal
    entries = []
    with open(journal_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    
    df = pd.DataFrame(entries)
    
    # Create animation
    create_animated_entropy_rhythm(
        df,
        save_path=save_path,
        fps=fps,
        normalize=normalize,
        title=title
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Plasticity Demo")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Hugging Face model name")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--cycles", type=int, default=5, help="Number of plasticity cycles")
    parser.add_argument("--steps_per_cycle", type=int, default=50, help="Training steps per cycle")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--run_stress_protocol", action="store_true", help="Run stress protocol after cycles")
    parser.add_argument("--stress_cycles", type=int, default=3, help="Number of stress cycles")
    
    args = parser.parse_args()
    
    run_neural_plasticity_demo(args)