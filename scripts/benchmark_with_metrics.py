#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark With Comprehensive Metrics

This script runs benchmarks with the comprehensive metrics collection system, tracking
pruning patterns, gate values, head importance, and performance metrics. It supports
both synthetic data and real data from Project Gutenberg books for realistic evaluation.

Usage:
    # Basic usage
    python scripts/benchmark_with_metrics.py --model_name gpt2 --output_dir ./benchmark_results
    
    # With real data from Project Gutenberg
    python scripts/benchmark_with_metrics.py --model_name distilgpt2 --eval_dataset gutenberg --use_real_data
    
    # Full benchmark with multiple pruning strategies and fine-tuning
    python scripts/benchmark_with_metrics.py \
      --model_name distilgpt2 \
      --output_dir ./benchmark_results \
      --pruning_strategies "random,entropy,magnitude" \
      --pruning_levels "0.1,0.3,0.5" \
      --learning_steps 100 \
      --learning_rate 2e-5 \
      --eval_dataset "gutenberg" \
      --use_real_data \
      --use_adaptive_lr
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def safe_update_tensor(tensor, new_value, index=None):
    """
    Safely update a tensor in-place, handling tensors that require gradients.
    
    Args:
        tensor: The tensor to update
        new_value: The new value to assign
        index: Optional index for updating specific elements
    """
    with torch.no_grad():
        if index is not None:
            # Update specific index
            tensor[index] = new_value
        else:
            # Update entire tensor or use copy_ for tensor-to-tensor assignment
            if isinstance(new_value, torch.Tensor) and tensor.size() == new_value.size():
                tensor.copy_(new_value)
            else:
                tensor.fill_(new_value)

def get_model_blocks(model):
    """
    Safely get the blocks from a model, handling different model structures.
    
    Args:
        model: The model to extract blocks from
        
    Returns:
        List of transformer blocks
    """
    # Check common model structures
    if hasattr(model, 'blocks'):
        return model.blocks
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
        return model.transformer.blocks
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'blocks'):
        return model.model.blocks
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'blocks'):
        return model.encoder.blocks
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        return model.encoder.layers
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'blocks'):
        return model.decoder.blocks
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        return model.decoder.layers
    elif hasattr(model, 'layers'):
        return model.layers
    else:
        # If we can't find blocks, print a warning and return an empty list
        print("WARNING: Could not find blocks in model structure. Check model compatibility.")
        print(f"Available attributes: {dir(model)}")
        return []

def get_attention_module(block):
    """
    Safely get the attention module from a block, handling different model structures.
    
    Args:
        block: The transformer block
        
    Returns:
        Attention module or None if not found
    """
    # Try dictionary-style access first (for ModuleDict)
    if isinstance(block, nn.ModuleDict) and "attn" in block:
        return block["attn"]
    elif isinstance(block, dict) and "attn" in block:
        return block["attn"]
    
    # Handle modules with nested attention
    attention_key_candidates = [
        "attn", "attention", "self_attention", "self_attn", 
        "mha", "multi_head_attention", "multihead_attn",
        "attention_layer", "q_attn"
    ]
    
    # Try direct attribute access first
    for key in attention_key_candidates:
        if hasattr(block, key):
            return getattr(block, key)
    
    # If we have a layernorm followed by attention structure
    if hasattr(block, "ln_1") and hasattr(block, "attn"):
        return block.attn
    if hasattr(block, "ln1") and hasattr(block, "attn"):
        return block.attn
    
    # If none of the known patterns match, try to find any attribute that might be an attention module
    for attr_name in dir(block):
        if "attention" in attr_name.lower() or "attn" in attr_name.lower():
            return getattr(block, attr_name)
    
    # If we can't find any attention-like module, print the available attributes and return None
    if isinstance(block, (nn.Module, nn.ModuleList, nn.ModuleDict)):
        print(f"WARNING: Could not find attention module in block with attributes: {dir(block)}")
    
    return None

def has_gate(attention_module):
    """
    Check if an attention module has a gate parameter.
    
    Args:
        attention_module: The attention module to check
        
    Returns:
        True if the module has a gate parameter, False otherwise
    """
    if attention_module is None:
        return False
    
    # Standard gate parameter
    if hasattr(attention_module, "gate"):
        return True
    
    # Check for head_gates parameter (some models use this name)
    if hasattr(attention_module, "head_gates"):
        return True
    
    # Check for gating_weights parameter
    if hasattr(attention_module, "gating_weights"):
        return True
        
    # Check if there are gate parameters in the state dict
    if hasattr(attention_module, "state_dict"):
        state_dict = attention_module.state_dict()
        
        # Look for any parameter with "gate" in the name
        for param_name in state_dict.keys():
            if "gate" in param_name.lower():
                return True
    
    return False


def get_gate_tensor(attention_module):
    """
    Get the gate tensor from an attention module.
    
    Args:
        attention_module: The attention module to get the gate from
        
    Returns:
        Gate tensor or None if not found
    """
    if attention_module is None:
        return None
    
    # Try standard gate parameter
    if hasattr(attention_module, "gate"):
        return attention_module.gate
    
    # Try head_gates parameter
    if hasattr(attention_module, "head_gates"):
        return attention_module.head_gates
    
    # Try gating_weights parameter
    if hasattr(attention_module, "gating_weights"):
        return attention_module.gating_weights
    
    # Try to find gate parameter in state dict
    if hasattr(attention_module, "state_dict"):
        state_dict = attention_module.state_dict()
        
        # Look for any parameter with "gate" in the name
        for param_name, param in state_dict.items():
            if "gate" in param_name.lower():
                # Get the parameter from the module
                for name, param in attention_module.named_parameters():
                    if name == param_name:
                        return param
    
    return None

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import necessary modules
from sentinel.utils.metric_collection import MetricCollector
from utils.evaluation import generate_text_samples, save_generated_samples, evaluate_text_coherence

# Import model loaders from sentinel namespace
try:
    # Try to use sentinel namespace first (preferred)
    from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model
    print("Using sentinel.models.loaders module")
except ImportError:
    # Fall back to original models.loaders if not available
    print("Warning: Using deprecated models.loaders module")
    from models.loaders.loader import load_baseline_model, load_adaptive_model

# Import pruning module
try:
    from sentinel.utils.pruning.pruning_module import PruningModule
except ImportError:
    from utils.pruning.pruning_module import PruningModule


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark with comprehensive metrics")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                       help="Name of the model to benchmark")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on (cpu, cuda)")
    
    # Pruning configuration
    parser.add_argument("--pruning_strategies", type=str, default="entropy,magnitude,random",
                       help="Comma-separated list of pruning strategies to benchmark")
    parser.add_argument("--pruning_levels", type=str, default="0.1,0.3,0.5",
                       help="Comma-separated list of pruning levels to benchmark")
    
    # Learning/fine-tuning configuration
    parser.add_argument("--learning_steps", type=int, default=0,
                       help="Number of learning steps after pruning (0 for no learning)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--early_stop_patience", type=int, default=5,
                       help="Number of evaluations with no improvement before early stopping")
    parser.add_argument("--eval_interval", type=int, default=50,
                       help="Evaluate model every N steps during fine-tuning")
    parser.add_argument("--use_adaptive_lr", action="store_true",
                       help="Use different learning rates for different parts of the model")
    
    # Evaluation configuration
    parser.add_argument("--eval_dataset", type=str, default=None,
                       help="Dataset for evaluation and fine-tuning. Options include: "
                            "'gutenberg' (Project Gutenberg books), 'pride' (Pride and Prejudice), "
                            "'sherlock' (Sherlock Holmes), 'monte' (Count of Monte Cristo), or "
                            "'processed' (pre-processed datasets from previous runs).")
    parser.add_argument("--eval_samples", type=int, default=100,
                       help="Number of evaluation samples")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation and fine-tuning")
    parser.add_argument("--use_real_data", action="store_true",
                       help="Use real data from Project Gutenberg or other sources instead of synthetic data. "
                            "When used with --eval_dataset, loads real text data from the specified source.")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose output")
    parser.add_argument("--save_checkpoints", action="store_true",
                       help="Save model checkpoints during fine-tuning")
    
    return parser.parse_args()


def prepare_model(args):
    """Prepare the model for benchmarking."""
    print(f"Loading baseline model: {args.model_name}")
    
    # Load baseline model
    baseline_model = load_baseline_model(args.model_name, args.device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create adaptive model
    print("Creating adaptive model")
    adaptive_model = load_adaptive_model(args.model_name, baseline_model, args.device, debug=args.verbose)
    
    return baseline_model, adaptive_model, tokenizer


def prepare_evaluation_data(tokenizer, args, split="validation"):
    """
    Prepare evaluation or training data.
    
    Args:
        tokenizer: The tokenizer to use
        args: Command line arguments
        split: Dataset split to use ('train' or 'validation')
        
    Returns:
        DataLoader for the specified dataset
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    if args.use_real_data and args.eval_dataset:
        # Load real data from files to avoid dataset loading issues
        texts = []
        
        # First, try to load data from local files in benchmark_data directory
        data_paths = {
            # Project Gutenberg books (real data)
            "gutenberg": "benchmark_data/gutenberg",
            "books": "benchmark_data/gutenberg",
            "classics": "benchmark_data/gutenberg",
            "literature": "benchmark_data/gutenberg",
            "novels": "benchmark_data/gutenberg",
            "pride": "benchmark_data/gutenberg/pride_and_prejudice.txt",
            "austen": "benchmark_data/gutenberg/pride_and_prejudice.txt",
            "sherlock": "benchmark_data/gutenberg/sherlock_holmes.txt",
            "holmes": "benchmark_data/gutenberg/sherlock_holmes.txt",
            "monte": "benchmark_data/gutenberg/count_of_monte_cristo.txt",
            "cristo": "benchmark_data/gutenberg/count_of_monte_cristo.txt",
            "dumas": "benchmark_data/gutenberg/count_of_monte_cristo.txt",
            
            # Pre-processed datasets (if available from previous runs)
            "processed": f"benchmark_data/gutenberg_processed_{split}.txt",
            "gutenberg_processed": f"benchmark_data/gutenberg_processed_{split}.txt",
            
            # Wikitext datasets (if available)
            "wikitext": "benchmark_data/wikitext-2-raw-v1-validation.txt",
            "wikitext-2": "benchmark_data/wikitext-2-raw-v1-validation.txt",
            "wikitext-103": "benchmark_data/wikitext-103-raw-v1-validation.txt",
        }
        
        # Determine which file to load based on dataset name
        data_file = None
        if args.eval_dataset:
            # Check for exact match first
            if args.eval_dataset in data_paths:
                data_file = data_paths[args.eval_dataset]
            else:
                # Try partial matches
                for key in data_paths:
                    if key in args.eval_dataset.lower():
                        data_file = data_paths[key]
                        break
        
        # Check for Gutenberg directory and download books if needed
        if data_file and 'gutenberg' in data_file and not os.path.exists(data_file):
            gutenberg_dir = os.path.dirname(data_file) if data_file.endswith('.txt') else data_file
            os.makedirs(gutenberg_dir, exist_ok=True)
            
            # Dictionary of essential Gutenberg books to download
            gutenberg_books = {
                "pride_and_prejudice.txt": "https://www.gutenberg.org/files/1342/1342-0.txt",
                "sherlock_holmes.txt": "https://www.gutenberg.org/files/1661/1661-0.txt",
                "count_of_monte_cristo.txt": "https://www.gutenberg.org/files/1184/1184-0.txt"
            }
            
            # Download missing books
            if not os.path.isdir(gutenberg_dir):
                os.makedirs(gutenberg_dir, exist_ok=True)
                
            for book_name, book_url in gutenberg_books.items():
                book_path = os.path.join(gutenberg_dir, book_name)
                if not os.path.exists(book_path):
                    print(f"Downloading {book_name} from Project Gutenberg...")
                    try:
                        import requests
                        response = requests.get(book_url)
                        with open(book_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded {book_name} successfully")
                    except Exception as e:
                        print(f"Error downloading {book_name}: {e}")
        
        # Try to load the data 
        if data_file and os.path.exists(data_file):
            try:
                # Handle directory case (multiple files)
                if os.path.isdir(data_file):
                    print(f"Loading real data from directory {data_file} for {split} split")
                    book_files = []
                    
                    # Get all text files in the directory
                    for filename in os.listdir(data_file):
                        if filename.endswith('.txt'):
                            book_files.append(os.path.join(data_file, filename))
                    
                    if not book_files:
                        print(f"No text files found in {data_file}")
                        texts = None
                    else:
                        # Choose different files for training and validation to prevent overlap
                        if split == "validation":
                            # Use 1/3 of books for validation
                            selected_files = book_files[:max(1, len(book_files) // 3)]
                            print(f"Selected {len(selected_files)} books for validation: {[os.path.basename(f) for f in selected_files]}")
                        else:
                            # Use 2/3 of books for training
                            selected_files = book_files[max(1, len(book_files) // 3):]
                            print(f"Selected {len(selected_files)} books for training: {[os.path.basename(f) for f in selected_files]}")
                        
                        # Load content from each selected file
                        for book_file in selected_files:
                            with open(book_file, 'r', encoding='utf-8') as f:
                                book_content = f.read()
                                
                                # Split book into paragraphs
                                paragraphs = book_content.split('\n\n')
                                
                                # Process paragraphs
                                for paragraph in paragraphs:
                                    paragraph = paragraph.strip().replace('\n', ' ')
                                    if len(paragraph) > 100:  # Keep only substantial paragraphs
                                        texts.append(paragraph)
                
                # Handle single file case
                else:
                    print(f"Loading real data from file {data_file} for {split} split")
                    with open(data_file, 'r', encoding='utf-8') as f:
                        # Read all content
                        content = f.read()
                        
                        # Split into paragraphs (both newline and double newline)
                        paragraphs = content.split('\n\n')
                        
                        # Process paragraphs
                        for paragraph in paragraphs:
                            paragraph = paragraph.strip().replace('\n', ' ')
                            if len(paragraph) > 100:  # Only keep substantial paragraphs
                                texts.append(paragraph)
                
                # If we have texts, we're done
                if texts:
                    # Remove Project Gutenberg headers and footers
                    texts = [t for t in texts if "PROJECT GUTENBERG" not in t.upper() 
                             and "GUTENBERG EBOOK" not in t.upper()
                             and "PRODUCED BY" not in t.upper()
                             and not t.startswith("***")]
                    
                    # Deduplicate texts
                    texts = list(set(texts))
                    
                    # Split texts that are too long into smaller segments
                    segmented_texts = []
                    for text in texts:
                        if len(text) > 500:  # Maximum segment length
                            # Split into sentences first
                            sentences = text.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
                            
                            # Recombine sentences into segments of appropriate length
                            current_segment = ""
                            for sentence in sentences:
                                if len(current_segment) + len(sentence) < 500:
                                    current_segment += sentence + " "
                                else:
                                    if current_segment:
                                        segmented_texts.append(current_segment.strip())
                                    current_segment = sentence + " "
                            
                            # Add the last segment if not empty
                            if current_segment:
                                segmented_texts.append(current_segment.strip())
                        else:
                            segmented_texts.append(text)
                    
                    texts = segmented_texts
                    
                    # Limit based on split type
                    if split == "validation":
                        # Shuffle and select validation samples
                        import random
                        random.shuffle(texts)
                        texts = texts[:min(len(texts), args.eval_samples)]
                    else:
                        # For training, use more samples but still limit
                        import random
                        random.shuffle(texts)
                        max_train_samples = min(len(texts), 5000)  # Limit to 5000 samples
                        texts = texts[:max_train_samples]
                    
                    print(f"Loaded {len(texts)} real text segments for {split}")
                    
                    # Save the processed dataset for future use
                    processed_file = f"benchmark_data/gutenberg_processed_{split}.txt"
                    with open(processed_file, 'w', encoding='utf-8') as f:
                        for text in texts[:min(len(texts), 1000)]:  # Save at most 1000 samples
                            f.write(text + "\n\n")
                    print(f"Saved processed dataset to {processed_file}")
                    
                else:
                    print(f"No valid texts found in {data_file}, falling back to synthetic data")
                    texts = None
            except Exception as e:
                print(f"Error loading data: {e}")
                texts = None
                
        # If still no texts, use high-quality synthetic data
        if not texts:
            print(f"Using high-quality synthetic data for {split} split")
            texts = []
            
            # Wikipedia article on "Artificial Intelligence"
            wiki_ai = """
            Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence of humans and other animals. Example tasks in which AI is applied include speech recognition, computer vision, translation between natural languages, and decision making.
            
            Some definitions of artificial intelligence focus on the use of technology to understand human intelligence, while others focus on solving problems. The field was founded on the assumption that human intelligence "can be so precisely described that a machine can be made to simulate it." This raised philosophical arguments about the mind and the ethical consequences of creating artificial beings endowed with human-like intelligence. These issues have been explored by myth, fiction, and philosophy since antiquity.
            
            Computer scientists and philosophers have suggested that strong AI may never be achieved due to the complexity of human intelligence, while others believe that artificial general intelligence, the ability of a machine to apply intelligence to any problem, is a reachable goal. 
            
            Since its beginning, AI research has explored symbol manipulation, neural networks, and methods based on statistics, probability, and economics. In the 1960s and 1970s, cybernetics and computational intelligence began to flourish, and in the 1990s and early 21st century, statistics-based machine learning achieved remarkable successes.
            """
            
            # Wikipedia article on "Neural Networks"
            wiki_nn = """
            Neural networks are a subset of machine learning and are at the heart of deep learning algorithms. Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.
            
            Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.
            
            Neural networks rely on training data to learn and improve their accuracy over time. However, once these learning algorithms are fine-tuned for accuracy, they become powerful tools in computer science and artificial intelligence, allowing us to classify and cluster data at a high velocity. Tasks in speech recognition or image recognition can take minutes versus hours when compared to the manual identification by human experts. One of the most well-known neural networks is Google's search algorithm.
            """
            
            # Wikipedia article on "Transformer Models"
            wiki_transformer = """
            A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing (NLP) and computer vision (CV).
            
            Like recurrent neural networks (RNNs), transformers are designed to process sequential input data, such as natural language, with applications extending to other tasks like text generation. However, unlike RNNs, transformers process the entire input all at once. The attention mechanism provides context for any position in the input sequence.
            
            The transformer was proposed in the paper "Attention Is All You Need" by researchers at Google Brain in 2017. It has become the model of choice for NLP problems, replacing RNN models such as long short-term memory (LSTM). The additional capability of transformers to process all inputs in parallel has reduced training times and enabled training on larger datasets, leading to models such as BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer).
            """
            
            # Creative writing from 20 top authors
            creative_texts = [
                "The old man was thin and gaunt with deep wrinkles in the back of his neck. The brown blotches of the benevolent skin cancer the sun brings from its reflection on the tropic sea were on his cheeks.",
                "All happy families are alike; each unhappy family is unhappy in its own way. Everything was in confusion in the Oblonskys' house.",
                "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity.",
                "For a long time, I went to bed early. Sometimes, my candle barely out, my eyes closed so quickly that I did not have time to tell myself: 'I'm falling asleep.'",
                "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions.",
                "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was to remember that distant afternoon when his father took him to discover ice.",
                "The sky above the port was the color of television, tuned to a dead channel. 'It's not like I'm using,' Case heard someone say, as he shouldered his way through the crowd.",
                "In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since. 'Whenever you feel like criticizing anyone,' he told me, 'just remember that all the people in this world haven't had the advantages that you've had.'",
                "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect. He was lying on his hard, as it were armor-plated, back and when he lifted his head a little he could see his dome-like brown belly.",
                "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.",
                "Lolita, light of my life, fire of my loins. My sin, my soul. Lo-lee-ta: the tip of the tongue taking a trip of three steps down the palate to tap, at three, on the teeth. Lo. Lee. Ta.",
                "I am an invisible man. No, I am not a spook like those who haunted Edgar Allan Poe; nor am I one of your Hollywood-movie ectoplasms. I am a man of substance, of flesh and bone, fiber and liquids—and I might even be said to possess a mind.",
                "Happy families are all alike; every unhappy family is unhappy in its own way. Everything was in confusion in the Oblonskys' house. The wife had discovered that the husband was carrying on an intrigue with a French girl, who had been a governess in their family.",
                "It was a pleasure to burn. It was a special pleasure to see things eaten, to see things blackened and changed. With the brass nozzle in his fists, with this great python spitting its venomous kerosene upon the world.",
                "Somewhere in la Mancha, in a place whose name I do not care to remember, a gentleman lived not long ago, one of those who has a lance and ancient shield on a shelf and keeps a skinny nag and a greyhound for racing.",
            ]
            
            # Scientific texts on physics and mathematics
            science_texts = [
                "The theory of relativity transformed our understanding of space and time. Einstein's equations demonstrated that space and time are not absolute, but rather form a four-dimensional spacetime that can be warped by matter and energy.",
                "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic scales. Its probabilistic nature challenged classical determinism, introducing concepts like wave-particle duality and quantum entanglement.",
                "The Riemann Hypothesis concerns the distribution of prime numbers and is considered one of the most important unsolved problems in mathematics. It states that all non-trivial zeros of the Riemann zeta function have real part 1/2.",
                "Machine learning algorithms learn patterns from data without explicit programming. Supervised learning uses labeled data for training, while unsupervised learning identifies patterns without labeled examples. Reinforcement learning involves agents learning through trial and error.",
                "The P versus NP problem is a major unsolved question in computer science. It asks whether every problem whose solution can be quickly verified by a computer can also be quickly solved by a computer. Its resolution would have profound implications for cryptography and optimization.",
                "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape from them. They form when massive stars collapse at the end of their life cycles. The event horizon marks the boundary beyond which escape is impossible.",
                "Fermat's Last Theorem states that no three positive integers a, b, and c can satisfy the equation aⁿ + bⁿ = cⁿ for any integer value of n greater than 2. It remained unproven for 358 years until Andrew Wiles presented a proof in 1994.",
                "The second law of thermodynamics states that the total entropy of an isolated system always increases over time. This fundamental principle explains why heat flows from hot to cold objects and why perpetual motion machines are impossible.",
                "String theory proposes that the fundamental constituents of reality are not point-like particles but tiny one-dimensional strings. Different vibration patterns of these strings correspond to different fundamental particles. The theory requires extra spatial dimensions beyond the familiar three.",
                "Neural networks are computational models inspired by the human brain. They consist of interconnected nodes or neurons organized in layers. Deep learning involves neural networks with many layers that can learn hierarchical representations of data.",
            ]
            
            # Combine all texts and create a diverse corpus
            all_texts = []
            all_texts.extend([wiki_ai, wiki_nn, wiki_transformer])
            all_texts.extend(creative_texts)
            all_texts.extend(science_texts)
            
            # Split longer texts into smaller segments
            for text in all_texts:
                # Split by paragraph or 300-character chunks for longer texts
                paragraphs = text.split("\n\n")
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if len(paragraph) > 50:  # Only keep substantial paragraphs
                        if len(paragraph) > 300:
                            # Split into smaller segments
                            for i in range(0, len(paragraph), 300):
                                segment = paragraph[i:i+300]
                                if len(segment.strip()) > 50:
                                    texts.append(segment)
                        else:
                            texts.append(paragraph)
            
            # Deduplicate and shuffle
            import random
            texts = list(set(texts))  # Remove duplicates
            random.shuffle(texts)     # Shuffle for more diversity
            
            # Split appropriately
            if split == "validation":
                texts = texts[:args.eval_samples]
            else:
                # Generate more training samples by combining texts if needed
                if len(texts) < 300:  # If we have too few unique samples
                    additional_texts = []
                    for _ in range(1000 // len(texts)):
                        for t1, t2 in zip(texts[::2], texts[1::2]):
                            combined = t1 + " " + t2
                            additional_texts.append(combined)
                    texts.extend(additional_texts)
                
                # Limit to a reasonable number
                texts = texts[:min(len(texts), 2000)]
            
            print(f"Created {len(texts)} diverse synthetic text segments for {split}")
            
            # Create directory for saving these synthetic datasets for future use
            os.makedirs("benchmark_data", exist_ok=True)
            
            # Also create gutenberg directory if it doesn't exist
            gutenberg_dir = "benchmark_data/gutenberg"
            if not os.path.exists(gutenberg_dir):
                os.makedirs(gutenberg_dir, exist_ok=True)
                
                # Download essential Gutenberg books
                gutenberg_books = {
                    "pride_and_prejudice.txt": "https://www.gutenberg.org/files/1342/1342-0.txt",
                    "sherlock_holmes.txt": "https://www.gutenberg.org/files/1661/1661-0.txt",
                    "count_of_monte_cristo.txt": "https://www.gutenberg.org/files/1184/1184-0.txt"
                }
                
                for book_name, book_url in gutenberg_books.items():
                    book_path = os.path.join(gutenberg_dir, book_name)
                    print(f"Downloading {book_name} from Project Gutenberg...")
                    try:
                        import requests
                        response = requests.get(book_url)
                        with open(book_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded {book_name} successfully")
                    except Exception as e:
                        print(f"Error downloading {book_name}: {e}")
            
            dataset_path = f"benchmark_data/synthetic_{split}.txt"
            with open(dataset_path, "w", encoding="utf-8") as f:
                for text in texts:
                    f.write(text + "\n\n")
            print(f"Saved synthetic dataset to {dataset_path} for future use")
    else:
        # Use synthetic data if real data not requested or no dataset provided
        texts = None
    
    # If no dataset or loading failed, create synthetic data
    if texts is None:
        print("Using synthetic data")
        
        # Create synthetic prompts
        prompts = [
            "The quick brown fox jumps over the lazy dog. The fox is known for its agility and speed.",
            "In a world where technology dominates, humans seek connection and meaning through art and nature.",
            "Once upon a time, there lived a wise king who ruled with compassion and justice. His kingdom prospered.",
            "The history of artificial intelligence dates back to ancient myths and legends about artificial beings.",
            "Climate change is affecting ecosystems worldwide, leading to rising sea levels and extreme weather.",
            "Scientists have discovered a new species of deep-sea creatures that can survive extreme pressure.",
            "The economic outlook remains uncertain as markets react to global political developments.",
            "Education has transformed dramatically in the digital age, with online learning becoming mainstream.",
            "Renewable energy sources are becoming increasingly competitive with traditional fossil fuels.",
            "Space exploration has entered a new era with private companies launching their own missions."
        ]
        
        # Generate longer texts for training
        if split == "train":
            extended_prompts = []
            for prompt in prompts:
                # Create variations with different continuations
                for i in range(10):
                    extended_prompts.append(f"{prompt} This is continuation {i} with additional text to provide more training data for the model.")
            prompts = extended_prompts
        
        # Repeat prompts to get enough samples
        texts = []
        while len(texts) < (args.eval_samples if split == "validation" else 1000):
            texts.extend(prompts)
        
        # Limit to the requested number of samples
        if split == "validation":
            texts = texts[:args.eval_samples]
        else:
            texts = texts[:1000]  # Use 1000 training samples for synthetic data
    
    # Tokenize texts
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt"
    )
    
    # Create appropriate labels for causal language modeling (shifted input_ids)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # Create labels for language modeling (shift input_ids right)
    labels = input_ids.clone()
    
    # Create dataloaders
    dataset = TensorDataset(input_ids, attention_mask, labels)
    
    # Use different shuffle settings for train vs. validation
    shuffle = (split == "train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    
    return dataloader


def finetune_model(model, train_dataloader, val_dataloader, tokenizer, collector, args, 
                strategy=None, pruning_level=None):
    """
    Fine-tune a model after pruning.
    
    Args:
        model: The model to fine-tune
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        tokenizer: Tokenizer for the model
        collector: MetricCollector instance
        args: Command line arguments
        strategy: Pruning strategy used (for logging)
        pruning_level: Pruning level used (for logging)
        
    Returns:
        Dictionary with training results
    """
    # Set model to training mode
    model.train()
    device = next(model.parameters()).device
    
    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Create optimizer with different parameter groups if requested
    if args.use_adaptive_lr:
        # Create parameter groups with different learning rates
        # Higher learning rate for attention heads to recover from pruning
        # Lower learning rate for embeddings and output layer
        
        # Get blocks and head parameters with higher learning rate
        blocks = get_model_blocks(model)
        head_params = []
        
        for block in blocks:
            attn_module = get_attention_module(block)
            if attn_module is not None:
                for param in attn_module.parameters():
                    head_params.append(param)
        
        # Special handling for different learning rates based on parameter type
        param_groups = []
        if head_params:  # Only add this group if we found attention head parameters
            param_groups.append({"params": head_params, "lr": args.learning_rate * 3.0})
            print(f"Found {len(head_params)} attention head parameters for higher learning rate")
        
        # Other parameters (default learning rate)
        other_params = []
        for name, param in model.named_parameters():
            # Skip parameters that are already in the head_params group
            if not any(param is hp for hp in head_params):
                other_params.append(param)
        
        param_groups.append({"params": other_params, "lr": args.learning_rate})
        
        optimizer = torch.optim.AdamW(param_groups)
        print(f"Using adaptive learning rates: {args.learning_rate * 3.0} for heads, {args.learning_rate} for other params")
    else:
        # Single learning rate for all parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        print(f"Using learning rate: {args.learning_rate}")
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.learning_steps, eta_min=args.learning_rate * 0.1
    )
    
    # Early stopping setup
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Training loop
    global_step = 0
    train_losses = []
    
    print(f"Starting fine-tuning for {args.learning_steps} steps "
          f"(eval every {args.eval_interval} steps, patience {args.early_stop_patience})")
    
    # Dictionary to store checkpoints
    checkpoints = {}
    
    start_time = time.time()
    
    while global_step < args.learning_steps:
        # Initialize progress bar for training
        progress_bar = tqdm(total=args.learning_steps, desc="Training", initial=global_step)
        
        # Loop through batches
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
            if global_step >= args.learning_steps:
                break
                
            # Move batch to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Create shifted targets for causal language modeling
            shift_labels = labels.clone()
            shift_labels[:, :-1] = labels[:, 1:]
            shift_labels[:, -1] = tokenizer.pad_token_id
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract logits
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Calculate loss
            shift_logits = logits[:, :-1, :]
            shift_targets = shift_labels[:, :-1]
            loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_targets.reshape(-1))
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update learning rate
            lr_scheduler.step()
            
            # Log training loss
            loss_val = loss.item()
            perplexity = torch.exp(loss).item()
            current_lr = lr_scheduler.get_last_lr()[0]
            train_losses.append(loss_val)
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss_val:.4f}",
                ppl=f"{perplexity:.1f}",
                lr=f"{current_lr:.2e}"
            )
            
            # Collect training metrics
            collector.collect_step_metrics(
                model=model,
                step=global_step,
                phase="train",
                inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                labels=shift_targets,
                logits=shift_logits,
                additional_metrics={
                    "train/loss": loss_val,
                    "train/perplexity": perplexity,
                    "train/lr": current_lr,
                    "train/strategy": strategy or "baseline",
                    "train/pruning_level": pruning_level or 0.0,
                    "train/batch": batch_idx,
                }
            )
            
            # Evaluate periodically
            if global_step % args.eval_interval == 0 or global_step == args.learning_steps - 1:
                val_metrics = evaluate_model(
                    model, val_dataloader, tokenizer, collector, global_step,
                    strategy=strategy, pruning_level=pruning_level
                )
                
                # Generate sample text to monitor quality during training
                progress_bar.write("\n🔍 PERIODIC TEXT GENERATION CHECK:")
                prompt = "Once upon a time"
                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                
                with torch.no_grad():
                    try:
                        model.eval()  # Ensure model is in eval mode
                        output = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_length=100,
                            temperature=0.7,
                            top_p=0.9,
                            num_return_sequences=1,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                        progress_bar.write(f"Step {global_step}, Prompt: '{prompt}'")
                        progress_bar.write(f"Generated: {generated_text[:200]}...\n")
                    except Exception as e:
                        progress_bar.write(f"Error generating text: {e}")
                
                # Save checkpoint if it's the best model so far
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    
                    # Update progress bar with validation metrics
                    progress_bar.set_postfix(
                        train_loss=f"{loss_val:.4f}",
                        val_loss=f"{val_metrics['loss']:.4f}",
                        val_ppl=f"{val_metrics['perplexity']:.1f}"
                    )
                    
                    # Save checkpoint if requested
                    if args.save_checkpoints:
                        checkpoint_path = os.path.join(
                            args.output_dir, 
                            f"{args.model_name.replace('/', '_')}_{strategy or 'baseline'}_{pruning_level or 0.0}_step_{global_step}.pt"
                        )
                        
                        # Save state dict only (more efficient than full model)
                        checkpoints[global_step] = {
                            "step": global_step,
                            "val_loss": val_metrics["loss"],
                            "val_perplexity": val_metrics["perplexity"],
                            "path": checkpoint_path
                        }
                        
                        torch.save(model.state_dict(), checkpoint_path)
                        progress_bar.write(f"✓ Saved checkpoint to {checkpoint_path}")
                else:
                    patience_counter += 1
                    progress_bar.write(f"⚠️ Validation loss did not improve. Patience: {patience_counter}/{args.early_stop_patience}")
                
                # Apply early stopping if patience is exceeded
                if patience_counter >= args.early_stop_patience:
                    progress_bar.write(f"⛔ Early stopping triggered after {global_step} steps")
                    break
                
                # Set model back to training mode after evaluation
                model.train()
            
            global_step += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Fine-tuning completed in {elapsed_time:.2f}s")
    
    # Calculate average training loss
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()
    
    print(f"Average training loss: {avg_train_loss:.4f}, Perplexity: {avg_train_perplexity:.4f}")
    
    # Final evaluation
    final_metrics = evaluate_model(
        model, val_dataloader, tokenizer, collector, global_step, 
        strategy=strategy, pruning_level=pruning_level, phase="final"
    )
    
    # Generate sample text to verify coherence
    # Use the modular text generation utility
    generated_samples = generate_text_samples(
        model=model,
        tokenizer=tokenizer,
        max_length=100,
        temperature=0.7,
        top_p=0.9
    )
    
    # Add coherence metrics
    evaluated_samples = evaluate_text_coherence(generated_samples)
    
    # Print a sample
    if evaluated_samples and "generated" in evaluated_samples[0]:
        sample = evaluated_samples[0]
        print(f"\nGenerated sample for '{sample['prompt']}':")
        print(f"{sample['generated'][:200]}...")
        if "metrics" in sample:
            print("\nMetrics:")
            for metric, value in sample["metrics"].items():
                print(f"- {metric}: {value:.4f}" if isinstance(value, float) else f"- {metric}: {value}")
    
    # Return results with generated samples and metrics
    return {
        "train_loss": avg_train_loss,
        "train_perplexity": avg_train_perplexity,
        "val_loss": final_metrics["loss"],
        "val_perplexity": final_metrics["perplexity"],
        "steps": global_step,
        "elapsed_time": elapsed_time,
        "best_val_loss": best_val_loss,
        "checkpoints": checkpoints,
        "generated_samples": evaluated_samples
    }


def evaluate_model(model, dataloader, tokenizer, collector, step, 
                  strategy=None, pruning_level=None, phase="eval"):
    """
    Evaluate model performance.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        tokenizer: Tokenizer for the model
        collector: MetricCollector instance
        step: Current step number
        strategy: Pruning strategy used (for logging)
        pruning_level: Pruning level used (for logging)
        phase: Evaluation phase name
        
    Returns:
        Dictionary with evaluation results
    """
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Track metrics
    all_losses = []
    all_perplexities = []
    batch_step = 0
    
    # Function to compute perplexity from loss
    def compute_perplexity(loss):
        return torch.exp(loss).item()
    
    # Run evaluation with progress bar
    progress_bar = tqdm(dataloader, desc=f"{phase.capitalize()} evaluation", leave=False)
    with torch.no_grad():
        for batch_num, (input_ids, attention_mask, labels) in enumerate(progress_bar):
            # Move batch to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Create shifted targets for causal language modeling
            shift_labels = labels.clone()
            shift_labels[:, :-1] = labels[:, 1:]
            shift_labels[:, -1] = tokenizer.pad_token_id
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract logits
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Calculate loss
            shift_logits = logits[:, :-1, :]
            shift_targets = shift_labels[:, :-1]
            loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_targets.reshape(-1))
            
            # Calculate perplexity
            perplexity = compute_perplexity(loss)
            
            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", ppl=f"{perplexity:.1f}")
            
            # Collect comprehensive metrics
            collector.collect_step_metrics(
                model=model,
                step=step + batch_step,
                phase=phase,
                inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                labels=shift_targets,
                logits=shift_logits,
                additional_metrics={
                    f"{phase}/batch": batch_num,
                    f"{phase}/loss": loss.item(),
                    f"{phase}/perplexity": perplexity,
                    f"{phase}/strategy": strategy or "baseline",
                    f"{phase}/pruning_level": pruning_level or 0.0
                }
            )
            
            # Store metrics
            all_losses.append(loss.item())
            all_perplexities.append(perplexity)
            
            batch_step += 1
    
    # Calculate average metrics
    avg_loss = sum(all_losses) / len(all_losses)
    avg_perplexity = sum(all_perplexities) / len(all_perplexities)
    
    print(f"{phase.capitalize()} Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}")
    
    return {
        "loss": avg_loss,
        "perplexity": avg_perplexity,
    }


def benchmark_model(model, dataloader, tokenizer, collector, args, strategy=None, pruning_level=None):
    """Benchmark a model with comprehensive metrics collection."""
    device = next(model.parameters()).device
    model.eval()
    
    # Initialize pruning if requested
    if strategy is not None and pruning_level is not None:
        print(f"\nBenchmarking with {strategy} pruning at level {pruning_level}")
        
        # Direct implementation of pruning for adaptive model
        try:
            # Get blocks and count total heads with progress indicator
            print("Analyzing model structure...")
            blocks = get_model_blocks(model)
            total_heads = 0
            head_info = []  # Store (layer_idx, attention_module, head_idx) for each head
            
            block_bar = tqdm(enumerate(blocks), total=len(blocks), desc="Scanning blocks", leave=False)
            for layer_idx, block in block_bar:
                attn_module = get_attention_module(block)
                if attn_module is not None and has_gate(attn_module):
                    gate_tensor = get_gate_tensor(attn_module)
                    if gate_tensor is not None:
                        num_heads = len(gate_tensor)
                        total_heads += num_heads
                        for head_idx in range(num_heads):
                            head_info.append((layer_idx, attn_module, head_idx))
                block_bar.set_postfix(heads=total_heads)
            
            # Apply simple pruning (set gates to 0)
            num_to_prune = int(total_heads * pruning_level)
            num_pruned = 0
            
            print(f"Pruning {num_to_prune} of {total_heads} heads using {strategy} strategy...")
            
            # Implement different pruning strategies
            if strategy == "random":
                # Random pruning
                import random
                
                # Shuffle and select heads to prune
                random.shuffle(head_info)
                prune_list = head_info[:num_to_prune]
                
                # Apply pruning using the safe update utility with progress bar
                prune_bar = tqdm(prune_list, desc="Pruning heads", leave=False)
                for layer_idx, attn_module, head_idx in prune_bar:
                    gate_tensor = get_gate_tensor(attn_module)
                    if gate_tensor is not None:
                        safe_update_tensor(gate_tensor, 0.0, index=head_idx)
                        num_pruned += 1
                        prune_bar.set_postfix(pruned=f"{num_pruned}/{num_to_prune}")
                    else:
                        prune_bar.write(f"Warning: Could not find gate tensor in layer {layer_idx}, head {head_idx}")
                    
            elif strategy == "magnitude":
                # Import the proper magnitude pruning implementation
                try:
                    print("Applying magnitude-based pruning using weight magnitudes...")
                    from sentinel.pruning.entropy_magnitude import magnitude_based_pruning
                    
                    # Get pruning indices
                    pruned_heads = magnitude_based_pruning(
                        model, 
                        prune_ratio=pruning_level, 
                        safe_update_tensor_fn=safe_update_tensor
                    )
                    num_pruned = len(pruned_heads)
                except Exception as e:
                    print(f"Error in magnitude pruning implementation: {e}")
                    print("Falling back to simple pruning...")
                    
                    # Fallback to simple pruning
                    remaining = num_to_prune
                    prune_bar = tqdm(head_info, desc="Pruning heads (fallback)", leave=False)
                    for layer_idx, attn_module, head_idx in prune_bar:
                        if remaining > 0:
                            gate_tensor = get_gate_tensor(attn_module)
                            if gate_tensor is not None:
                                safe_update_tensor(gate_tensor, 0.0, index=head_idx)
                                num_pruned += 1
                                remaining -= 1
                                prune_bar.set_postfix(pruned=f"{num_pruned}/{num_to_prune}")
                            else:
                                prune_bar.write(f"Warning: Could not find gate tensor in layer {layer_idx}, head {head_idx}")
                
            elif strategy == "entropy":
                # Import the proper entropy pruning implementation
                try:
                    print("Collecting attention distributions for entropy-based pruning...")
                    from sentinel.pruning.entropy_magnitude import (
                        collect_attention_distributions,
                        entropy_based_pruning
                    )
                    
                    # Collect attention distributions (sample a few batches)
                    distributions = collect_attention_distributions(
                        model,
                        dataloader,
                        num_batches=5  # Adjust based on dataset size
                    )
                    
                    # Apply entropy-based pruning
                    pruned_heads = entropy_based_pruning(
                        model,
                        distributions,
                        prune_ratio=pruning_level,
                        safe_update_tensor_fn=safe_update_tensor
                    )
                    num_pruned = len(pruned_heads)
                except Exception as e:
                    print(f"Error in entropy pruning implementation: {e}")
                    print("Falling back to simple pruning...")
                    
                    # Fallback to simple pruning
                    remaining = num_to_prune
                    prune_bar = tqdm(head_info, desc="Pruning heads (fallback)", leave=False)
                    for layer_idx, attn_module, head_idx in prune_bar:
                        if remaining > 0:
                            gate_tensor = get_gate_tensor(attn_module)
                            if gate_tensor is not None:
                                safe_update_tensor(gate_tensor, 0.0, index=head_idx)
                                num_pruned += 1
                                remaining -= 1
                                prune_bar.set_postfix(pruned=f"{num_pruned}/{num_to_prune}")
                            else:
                                prune_bar.write(f"Warning: Could not find gate tensor in layer {layer_idx}, head {head_idx}")
            
            print(f"✓ Pruned {num_pruned}/{total_heads} heads ({num_pruned/total_heads:.1%})")
        
        except Exception as e:
            print(f"Error applying pruning: {e}")
            return {
                "loss": float('nan'),
                "perplexity": float('nan'),
                "elapsed_time": 0,
            }
    else:
        print("\nBenchmarking baseline model (no pruning)")
    
    # Run evaluation
    start_time = time.time()
    
    # Generate a quick sample immediately to see baseline generation before any training
    if args.verbose:
        print("\n🔍 BASELINE TEXT GENERATION BEFORE FINE-TUNING:")
        prompt = "Once upon a time"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            try:
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=100,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"\nPrompt: {prompt}")
                print(f"Generated: {generated_text}\n")
            except Exception as e:
                print(f"Error generating text: {e}")
    
    # Evaluate the model
    metrics = evaluate_model(
        model, dataloader, tokenizer, collector, 0,
        strategy=strategy, pruning_level=pruning_level
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Add elapsed time to metrics
    metrics["elapsed_time"] = elapsed_time
    
    print(f"Evaluation completed in {elapsed_time:.2f}s")
    
    # Fine-tune if requested
    if args.learning_steps > 0:
        print(f"\nFine-tuning model for {args.learning_steps} steps...")
        
        # Prepare training data
        train_dataloader = prepare_evaluation_data(tokenizer, args, split="train")
        
        # Fine-tune the model
        train_results = finetune_model(
            model, train_dataloader, dataloader, tokenizer, collector, args,
            strategy=strategy, pruning_level=pruning_level
        )
        
        # Update metrics with training results
        metrics.update(train_results)
        
        print(f"\nFinal metrics after fine-tuning:")
        print(f"  Initial loss: {metrics['loss']:.4f}, perplexity: {metrics['perplexity']:.4f}")
        print(f"  Final loss: {metrics['val_loss']:.4f}, perplexity: {metrics['val_perplexity']:.4f}")
        
        # Calculate improvement
        loss_improvement = (metrics['loss'] - metrics['val_loss']) / metrics['loss'] * 100
        ppl_improvement = (metrics['perplexity'] - metrics['val_perplexity']) / metrics['perplexity'] * 100
        
        print(f"  Improvement: Loss: {loss_improvement:.2f}%, Perplexity: {ppl_improvement:.2f}%")
    
    return metrics


def main():
    """Main function."""
    args = setup_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare model
    baseline_model, adaptive_model, tokenizer = prepare_model(args)
    
    # Prepare evaluation data
    eval_dataloader = prepare_evaluation_data(tokenizer, args, split="validation")
    
    # Initialize metric collector
    collector = MetricCollector(
        output_dir=output_dir,
        model_name=args.model_name,
        track_gate_values=True,
        track_head_metrics=True,
        track_performance=True,
        track_pruning_patterns=True,
        compare_with_static=True,
        log_level="INFO"
    )
    
    # Parse pruning configurations
    pruning_strategies = args.pruning_strategies.split(",")
    pruning_levels = [float(level) for level in args.pruning_levels.split(",")]
    
    # Save benchmark configuration
    config = {
        "model_name": args.model_name,
        "device": args.device,
        "max_length": args.max_length,
        "eval_dataset": args.eval_dataset,
        "eval_samples": args.eval_samples,
        "batch_size": args.batch_size,
        "pruning_strategies": pruning_strategies,
        "pruning_levels": pruning_levels,
        "learning_steps": args.learning_steps,
        "learning_rate": args.learning_rate,
        "early_stop_patience": args.early_stop_patience,
        "eval_interval": args.eval_interval,
        "use_adaptive_lr": args.use_adaptive_lr,
        "use_real_data": args.use_real_data,
        "timestamp": timestamp
    }
    
    with open(os.path.join(output_dir, "benchmark_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Start with baseline (no pruning)
    print("\n" + "="*50)
    print(f"Benchmarking model: {args.model_name}")
    print("="*50)
    
    baseline_results = benchmark_model(
        model=adaptive_model,
        dataloader=eval_dataloader,
        tokenizer=tokenizer,
        collector=collector,
        args=args
    )
    
    # Register baseline results as a "strategy" for comparison
    collector.register_static_pruning_metrics("baseline", baseline_results)
    
    # Run benchmarks for each pruning strategy and level
    all_results = {
        "baseline": baseline_results
    }
    
    # Create progress bars for overall progress
    strategy_bar = tqdm(pruning_strategies, desc="Pruning Strategies", position=0)
    
    for strategy in strategy_bar:
        strategy_results = {}
        strategy_bar.set_description(f"Strategy: {strategy}")
        
        # Add a nested progress bar for levels
        level_bar = tqdm(pruning_levels, desc="Pruning Levels", position=1, leave=False)
        
        for level in level_bar:
            level_bar.set_description(f"Level: {level}")
            print("\n" + "="*50)
            print(f"Benchmarking {strategy} pruning at level {level}")
            print("="*50)
            
            # Clone model to avoid interference between runs
            model_clone = load_adaptive_model(args.model_name, baseline_model, args.device, debug=False)
            
            # Run benchmark
            results = benchmark_model(
                model=model_clone,
                dataloader=eval_dataloader,
                tokenizer=tokenizer,
                collector=collector,
                args=args,
                strategy=strategy,
                pruning_level=level
            )
            
            strategy_results[str(level)] = results
            
            # Register results for comparison
            if 'val_perplexity' in results:
                # Use fine-tuned metrics for comparison if available
                collector.register_static_pruning_metrics(
                    f"{strategy}_{level}", 
                    {"loss": results["val_loss"], "perplexity": results["val_perplexity"]}
                )
            else:
                # Use raw evaluation metrics
                collector.register_static_pruning_metrics(f"{strategy}_{level}", results)
            
            # Clear memory
            del model_clone
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        all_results[strategy] = strategy_results
    
    # Save all results
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        # Convert any non-serializable objects
        def sanitize_for_json(obj):
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items() if k != 'checkpoints'}
            elif isinstance(obj, list):
                return [sanitize_for_json(v) for v in obj]
            elif isinstance(obj, (torch.Tensor, np.ndarray)):
                return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
            else:
                return obj
            
        sanitized_results = sanitize_for_json(all_results)
        json.dump(sanitized_results, f, indent=2)
    
    # Save sample generated texts to a separate file for easy viewing
    print("\n" + "="*50)
    print("Saving generated text samples")
    print("="*50)
    
    generated_samples_all = {}
    for strategy, levels in all_results.items():
        if strategy == "baseline":
            if "generated_samples" in levels:
                generated_samples_all["baseline"] = levels["generated_samples"]
        else:
            for level, results in levels.items():
                if "generated_samples" in results:
                    generated_samples_all[f"{strategy}_{level}"] = results["generated_samples"]
    
    # Use the modular utility to save samples
    save_generated_samples(
        samples_dict=generated_samples_all,
        output_path=os.path.join(output_dir, "generated_samples.txt"),
        title="GENERATED TEXT SAMPLES - BENCHMARK RESULTS"
    )
    
    # Generate comprehensive analysis
    print("\n" + "="*50)
    print("Generating comprehensive analysis")
    print("="*50)
    
    report = collector.generate_report()
    
    # Create visualizations
    collector.visualize_metrics()
    
    # Save metrics to CSV for external analysis
    collector.save_metrics_csv()
    
    print("\n" + "="*50)
    print(f"Benchmark complete. Results saved to {output_dir}")
    print(f"Generated text samples saved to {os.path.join(output_dir, 'generated_samples.txt')}")
    print("="*50)
    
    # Print summary of best strategies
    if "static_comparison" in report and "overall_winner" in report["static_comparison"]:
        winner = report["static_comparison"]["overall_winner"]
        print(f"\nOverall best strategy: {winner}")
        
        if args.learning_steps > 0:
            print("\nFine-tuning summary:")
            for strategy in pruning_strategies:
                for level in pruning_levels:
                    results = all_results[strategy][str(level)]
                    if 'val_perplexity' in results:
                        ppl_improvement = ((results['perplexity'] - results['val_perplexity']) / 
                                         results['perplexity'] * 100)
                        print(f"  {strategy} at {level}: Initial PPL {results['perplexity']:.2f} → "
                              f"Final PPL {results['val_perplexity']:.2f} ({ppl_improvement:+.2f}%)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())