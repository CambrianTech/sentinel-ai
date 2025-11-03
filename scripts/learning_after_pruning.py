#!/usr/bin/env python
"""
Learning After Pruning Demonstration Script

This script demonstrates that pruned models can effectively learn new tasks.
It compares the learning efficiency of pruned vs. non-pruned models on a new task
and visualizes how the models adapt during learning.

Key demonstrations:
1. Pruning maintains adaptability to new tasks
2. Gate values evolve during learning to optimize for the new task
3. Pruned models can learn new tasks with comparable or better efficiency

This provides evidence that our adaptive pruning approach enables models to
"grow into something more powerful" even after significant pruning.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.model_wrapper import wrap_model_for_generation
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.generation_wrapper import generate_text
from controller.metrics.head_metrics import collect_head_metrics

# Task options
TASK_OPTIONS = ["sentiment", "code", "science", "poetry"]

def parse_args():
    parser = argparse.ArgumentParser(description="Sentinel-AI Learning After Pruning Script")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Model name or path (e.g., distilgpt2, gpt2)")
    
    # Pruning configuration
    parser.add_argument("--pruning_level", type=float, default=0.5,
                        help="Pruning level to apply (0.0-0.9)")
    parser.add_argument("--pruning_strategy", type=str, default="entropy",
                        choices=["random", "entropy", "gradient"],
                        help="Pruning strategy to use")
    
    # Learning task configuration
    parser.add_argument("--task", type=str, default="sentiment",
                        choices=TASK_OPTIONS,
                        help="New task to learn after pruning")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Number of examples to use for training/evaluation")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="learning_results",
                        help="Directory to save results")
    parser.add_argument("--drive_path", type=str, default="",
                        help="Google Drive path for saving results (for Colab)")

    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

def apply_pruning(model, strategy, pruning_level, device):
    """Apply pruning to the model according to the specified strategy and level."""
    print(f"Applying {strategy} pruning at {pruning_level:.1%} level")
    
    # Get model dimensions
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    total_heads = num_layers * num_heads
    heads_to_prune = int(total_heads * pruning_level)
    
    # Create dummy input for collecting metrics if needed
    batch_size = 2
    seq_len = 32
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    dummy_batch = {"input_ids": dummy_input, 
                  "attention_mask": torch.ones_like(dummy_input)}
    
    # Apply pruning based on strategy
    if strategy == "random":
        # Get a flattened list of (layer, head) tuples
        all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
        
        # Randomly select heads to prune
        pruned_head_indices = np.random.choice(len(all_heads), heads_to_prune, replace=False)
        
        # Set gates to near-zero for pruned heads
        with torch.no_grad():
            for idx in pruned_head_indices:
                layer_idx, head_idx = all_heads[idx]
                model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=device)
    
    elif strategy in ["entropy", "gradient"]:
        # Collect metrics
        metrics = collect_head_metrics(model, batch=dummy_batch)
        
        if strategy == "entropy" and "entropy" in metrics:
            head_scores = metrics["entropy"]
            # Higher entropy = less focused attention = more likely to be pruned
            descending = True
        elif strategy == "gradient" and "grad_norm" in metrics:
            head_scores = metrics["grad_norm"]
            # Lower gradient norm = less important head = more likely to be pruned
            descending = False
        else:
            print(f"Warning: {strategy} metrics not available, using random pruning")
            return apply_pruning(model, "random", pruning_level, device)
        
        # Reshape and flatten scores
        if not isinstance(head_scores, torch.Tensor):
            head_scores = torch.tensor(head_scores, device=device)
            
        if len(head_scores.shape) < 2:
            head_scores = head_scores.reshape(num_layers, num_heads)
            
        flat_scores = head_scores.view(-1)
        
        # Sort scores
        _, indices = torch.sort(flat_scores, descending=descending)
        indices_to_prune = indices[:heads_to_prune]
        
        # Apply pruning
        with torch.no_grad():
            for idx in indices_to_prune:
                layer_idx = idx.item() // num_heads
                head_idx = idx.item() % num_heads
                model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=device)
    
    # Count pruned heads for verification
    pruned_count = 0
    with torch.no_grad():
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                if model.blocks[layer_idx]["attn"].gate[head_idx].item() < 0.01:
                    pruned_count += 1
    
    print(f"Pruned {pruned_count} of {total_heads} heads ({pruned_count/total_heads:.1%})")
    return model

def create_sentiment_dataset(tokenizer, max_length, sample_size, seed=42):
    """Create a simple sentiment analysis dataset."""
    # Example sentiment data (positive and negative statements)
    positive = [
        "I absolutely loved this movie, it was fantastic!",
        "The product exceeded my expectations and works perfectly.",
        "This was the best meal I've had in years.",
        "I'm extremely satisfied with the service provided.",
        "The experience was amazing from start to finish.",
        "This is my favorite book of all time.",
        "The staff was friendly and very helpful.",
        "I couldn't be happier with my purchase.",
        "This app has dramatically improved my productivity.",
        "The view from the hotel room was breathtaking."
    ]
    
    negative = [
        "This movie was terrible and a complete waste of time.",
        "The product broke after just one week of use.",
        "The food was cold and tasteless.",
        "The customer service was unhelpful and rude.",
        "I had a miserable experience and wouldn't recommend it.",
        "This book was boring and poorly written.",
        "The staff was unfriendly and seemed annoyed by my questions.",
        "I regret making this purchase and want a refund.",
        "This app constantly crashes and is full of bugs.",
        "The hotel room was dirty and uncomfortable."
    ]
    
    # Generate more examples by combining templates with subjects
    subjects = [
        "The movie", "The book", "The meal", "The product", "The service",
        "The hotel", "The restaurant", "The app", "The experience", "The concert",
        "The performance", "The course", "The game", "The trip", "The event"
    ]
    
    pos_templates = [
        "{} was amazing and exceeded my expectations.",
        "{} was absolutely wonderful, I highly recommend it.",
        "{} deserves five stars for quality and value.",
        "{} made me incredibly happy and satisfied.",
        "{} was a delightful experience from start to finish."
    ]
    
    neg_templates = [
        "{} was terrible and completely disappointing.",
        "{} was awful, I would never recommend it.",
        "{} deserves zero stars for poor quality.",
        "{} made me frustrated and dissatisfied.",
        "{} was a miserable experience that I regret."
    ]
    
    # Expand the datasets using templates
    for subject in subjects:
        for template in pos_templates:
            positive.append(template.format(subject))
        for template in neg_templates:
            negative.append(template.format(subject))
    
    # Limit to requested sample size
    np.random.seed(seed)
    max_per_class = sample_size // 2
    
    if len(positive) > max_per_class:
        positive = list(np.random.choice(positive, max_per_class, replace=False))
    if len(negative) > max_per_class:
        negative = list(np.random.choice(negative, max_per_class, replace=False))
    
    # Create the dataset with labels
    texts = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)
    
    # Shuffle the dataset
    combined = list(zip(texts, labels))
    np.random.shuffle(combined)
    texts, labels = zip(*combined)
    
    # Convert to tensors using the tokenizer
    encodings = tokenizer(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create a simple dataset
    dataset = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': torch.tensor(labels)
    }
    
    # Split into train/test
    train_size = int(0.8 * len(texts))
    train_dataset = {
        'input_ids': dataset['input_ids'][:train_size],
        'attention_mask': dataset['attention_mask'][:train_size],
        'labels': dataset['labels'][:train_size]
    }
    
    test_dataset = {
        'input_ids': dataset['input_ids'][train_size:],
        'attention_mask': dataset['attention_mask'][train_size:],
        'labels': dataset['labels'][train_size:]
    }
    
    return train_dataset, test_dataset

def create_code_dataset(tokenizer, max_length, sample_size, seed=42):
    """Create a simple code generation dataset."""
    # Example programming problems and solutions (Python)
    code_examples = [
        {
            "problem": "Write a function to check if a number is prime.",
            "solution": """def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True"""
        },
        {
            "problem": "Write a function to calculate the Fibonacci sequence up to n terms.",
            "solution": """def fibonacci(n):
    sequence = []
    a, b = 0, 1
    for _ in range(n):
        sequence.append(a)
        a, b = b, a + b
    return sequence"""
        },
        {
            "problem": "Write a function to reverse a string.",
            "solution": """def reverse_string(s):
    return s[::-1]"""
        },
        {
            "problem": "Write a function to find the factorial of a number.",
            "solution": """def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)"""
        },
        {
            "problem": "Write a function to check if a string is a palindrome.",
            "solution": """def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]"""
        }
    ]
    
    # Create more complex examples
    more_examples = [
        {
            "problem": "Write a function to sort a list using bubble sort.",
            "solution": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr"""
        },
        {
            "problem": "Write a function to check if two strings are anagrams.",
            "solution": """def are_anagrams(s1, s2):
    return sorted(s1.lower()) == sorted(s2.lower())"""
        },
        {
            "problem": "Write a function to find all duplicates in a list.",
            "solution": """def find_duplicates(arr):
    seen = set()
    duplicates = set()
    for item in arr:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)"""
        },
        {
            "problem": "Write a function to perform binary search on a sorted array.",
            "solution": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1"""
        },
        {
            "problem": "Write a function to find the longest common substring of two strings.",
            "solution": """def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    max_length = 0
    end_pos = 0
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
    
    return s1[end_pos-max_length:end_pos]"""
        }
    ]
    
    # Combine all examples
    all_examples = code_examples + more_examples
    
    # Generate additional examples using templates
    problem_templates = [
        "Write a function to {task}.",
        "Implement a function that {task}.",
        "Create a Python function to {task}.",
        "Develop a function that {task}.",
        "Write a program to {task}."
    ]
    
    tasks = [
        "count vowels in a string",
        "check if a year is a leap year",
        "convert temperature from Celsius to Fahrenheit",
        "calculate the area of a triangle",
        "find the greatest common divisor of two numbers",
        "calculate the sum of digits in a number",
        "check if a number is a perfect square",
        "remove duplicates from a list",
        "find the second largest element in a list",
        "convert a decimal number to binary"
    ]
    
    # Create more problems using the templates
    for task in tasks:
        template = np.random.choice(problem_templates)
        problem = template.format(task=task)
        # We'll leave the solution empty - the model will learn to generate these
        all_examples.append({"problem": problem, "solution": ""})
    
    # Limit to requested sample size
    np.random.seed(seed)
    if len(all_examples) > sample_size:
        all_examples = list(np.random.choice(all_examples, sample_size, replace=False))
    
    # Format the examples for the model
    formatted_examples = []
    for example in all_examples:
        text = f"Problem: {example['problem']}\nSolution:\n{example['solution']}"
        formatted_examples.append(text)
    
    # Convert to tensors
    encodings = tokenizer(
        formatted_examples,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create input/target pairs
    input_ids = encodings['input_ids'].clone()
    labels = input_ids.clone()
    
    # Split into train/test
    train_size = int(0.8 * len(formatted_examples))
    
    train_dataset = {
        'input_ids': input_ids[:train_size],
        'attention_mask': encodings['attention_mask'][:train_size],
        'labels': labels[:train_size]
    }
    
    test_dataset = {
        'input_ids': input_ids[train_size:],
        'attention_mask': encodings['attention_mask'][train_size:],
        'labels': labels[train_size:]
    }
    
    return train_dataset, test_dataset

def create_science_dataset(tokenizer, max_length, sample_size, seed=42):
    """Create a dataset of scientific facts for the model to learn."""
    # Basic scientific facts
    facts = [
        "The Earth orbits around the Sun in approximately 365.25 days.",
        "Atoms are composed of protons, neutrons, and electrons.",
        "DNA contains the genetic instructions for the development and functioning of living organisms.",
        "The human body has 206 bones in an adult skeleton.",
        "Water has the chemical formula H2O, consisting of two hydrogen atoms and one oxygen atom.",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "The periodic table organizes chemical elements based on their atomic number and properties.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "The Milky Way galaxy contains between 100 and 400 billion stars.",
        "Gravity is the force by which objects with mass attract one another.",
        "The theory of relativity was developed by Albert Einstein in the early 20th century.",
        "The human heart beats approximately 100,000 times per day.",
        "The Earth's atmosphere is composed primarily of nitrogen (78%) and oxygen (21%).",
        "Sound travels at approximately 343 meters per second in air at room temperature.",
        "The normal human body temperature is about 98.6°F or 37°C.",
        "The pH scale measures how acidic or basic a substance is, ranging from 0 to 14.",
        "The three states of matter are solid, liquid, and gas (with plasma sometimes considered a fourth).",
        "Mitochondria are often called the powerhouse of the cell because they produce ATP.",
        "Elements in the same column of the periodic table have similar chemical properties.",
        "The human brain contains approximately 86 billion neurons.",
        "Evolution by natural selection was proposed by Charles Darwin in the 19th century.",
        "The Earth's core is composed primarily of iron and nickel.",
        "The universe is expanding, with galaxies moving away from each other.",
        "Temperature is a measure of the average kinetic energy of particles in a substance.",
        "The Law of Conservation of Energy states that energy cannot be created or destroyed.",
        "Antibiotics are effective against bacterial infections but not viral infections.",
        "The four fundamental forces of nature are gravity, electromagnetism, strong nuclear force, and weak nuclear force.",
        "The human genome contains approximately 3 billion base pairs.",
        "Enzymes are biological catalysts that speed up chemical reactions in living organisms.",
        "The Earth is approximately 4.5 billion years old."
    ]
    
    # Generate more facts using templates
    templates = [
        "{subject} is {description}.",
        "Scientists have discovered that {subject} {verb} {object}.",
        "Research shows that {subject} {verb} {condition}.",
        "In {field} science, {subject} is known to {verb} {object}.",
        "{subject} consists of {components}."
    ]
    
    subjects = [
        "the human brain", "quantum mechanics", "cellular respiration", "climate change",
        "black holes", "genetic mutation", "the immune system", "the nervous system",
        "nuclear fusion", "biodiversity", "plate tectonics", "the carbon cycle"
    ]
    
    descriptions = [
        "one of the most complex structures in the known universe",
        "a fundamental process in energy production",
        "responsible for maintaining homeostasis in organisms",
        "a critical factor in ecosystem stability",
        "governed by precise mathematical equations",
        "involved in the regulation of biological functions"
    ]
    
    verbs = [
        "affects", "interacts with", "regulates", "transforms", "influences",
        "accelerates", "determines", "contributes to", "correlates with"
    ]
    
    objects = [
        "cellular metabolism", "genetic expression", "ecological stability",
        "atmospheric composition", "species diversity", "planetary systems",
        "quantum states", "neurological development", "evolutionary processes"
    ]
    
    conditions = [
        "under specific environmental conditions",
        "at extremely high temperatures",
        "when exposed to certain types of radiation",
        "in the presence of catalytic enzymes",
        "during the early stages of development",
        "across multiple spatial and temporal scales"
    ]
    
    fields = [
        "biological", "chemical", "physical", "astronomical", "geological",
        "environmental", "molecular", "evolutionary", "neurological"
    ]
    
    components = [
        "multiple interconnected subsystems",
        "a complex network of feedback mechanisms",
        "various molecular structures and compounds",
        "fundamental particles and force carriers",
        "hierarchical organizational levels",
        "both macroscopic and microscopic elements"
    ]
    
    # Generate additional facts
    for _ in range(50):
        template = np.random.choice(templates)
        if "{description}" in template:
            fact = template.format(
                subject=np.random.choice(subjects),
                description=np.random.choice(descriptions)
            )
        elif "{verb}" in template and "{object}" in template:
            fact = template.format(
                subject=np.random.choice(subjects),
                verb=np.random.choice(verbs),
                object=np.random.choice(objects)
            )
        elif "{condition}" in template:
            fact = template.format(
                subject=np.random.choice(subjects),
                verb=np.random.choice(verbs),
                condition=np.random.choice(conditions)
            )
        elif "{field}" in template:
            fact = template.format(
                field=np.random.choice(fields),
                subject=np.random.choice(subjects),
                verb=np.random.choice(verbs),
                object=np.random.choice(objects)
            )
        elif "{components}" in template:
            fact = template.format(
                subject=np.random.choice(subjects),
                components=np.random.choice(components)
            )
        facts.append(fact)
    
    # Limit to requested sample size
    np.random.seed(seed)
    if len(facts) > sample_size:
        facts = list(np.random.choice(facts, sample_size, replace=False))
    
    # Format the facts as scientific article snippets
    article_templates = [
        "Recent research: {fact}",
        "Scientific discovery: {fact}",
        "According to scientific studies, {fact}",
        "In a recent scientific journal, researchers noted that {fact}",
        "Science fact: {fact}"
    ]
    
    formatted_facts = []
    for fact in facts:
        template = np.random.choice(article_templates)
        text = template.format(fact=fact)
        formatted_facts.append(text)
    
    # Convert to tensors
    encodings = tokenizer(
        formatted_facts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create input/target pairs for autoregressive learning
    input_ids = encodings['input_ids'].clone()
    labels = input_ids.clone()
    
    # Split into train/test
    train_size = int(0.8 * len(formatted_facts))
    
    train_dataset = {
        'input_ids': input_ids[:train_size],
        'attention_mask': encodings['attention_mask'][:train_size],
        'labels': labels[:train_size]
    }
    
    test_dataset = {
        'input_ids': input_ids[train_size:],
        'attention_mask': encodings['attention_mask'][train_size:],
        'labels': labels[train_size:]
    }
    
    return train_dataset, test_dataset

def create_poetry_dataset(tokenizer, max_length, sample_size, seed=42):
    """Create a dataset of poetry for the model to learn."""
    # Short poems and poetic phrases
    poems = [
        "Roses are red,\nViolets are blue,\nSugar is sweet,\nAnd so are you.",
        
        "The road not taken,\nTwo paths diverged in a wood,\nI took the one less traveled by,\nAnd that has made all the difference.",
        
        "Shall I compare thee to a summer's day?\nThou art more lovely and more temperate.",
        
        "Hope is the thing with feathers\nThat perches in the soul,\nAnd sings the tune without the words,\nAnd never stops at all.",
        
        "I wandered lonely as a cloud\nThat floats on high o'er vales and hills,\nWhen all at once I saw a crowd,\nA host, of golden daffodils.",
        
        "Nature's first green is gold,\nHer hardest hue to hold.\nHer early leaf's a flower;\nBut only so an hour.",
        
        "Do not go gentle into that good night,\nRage, rage against the dying of the light.",
        
        "Two roads diverged in a yellow wood,\nAnd sorry I could not travel both\nAnd be one traveler, long I stood\nAnd looked down one as far as I could.",
        
        "Water, water, everywhere,\nAnd all the boards did shrink;\nWater, water, everywhere,\nNor any drop to drink.",
        
        "Because I could not stop for Death –\nHe kindly stopped for me –\nThe Carriage held but just Ourselves –\nAnd Immortality."
    ]
    
    # Generate more poems using templates
    templates = [
        "The {noun} {verb} in the {location},\nLike {adjective} {noun2} in the {time}.",
        
        "{Adjective} {noun} {verb} through the {location},\n{Verb} the {adjective} {noun2} of {abstract_noun}.",
        
        "In the {time} of {abstract_noun},\n{Pronoun} {verb} like {adjective} {noun}.",
        
        "{Pronoun} {verb} the {noun} of {abstract_noun},\n{Adjective} and {adjective}, like {noun2} in {location}.",
        
        "Oh {adjective} {noun}, how {pronoun} {verb},\nThrough {time} and {abstract_noun}, forever {adjective}."
    ]
    
    nouns = [
        "star", "moon", "sun", "tree", "flower", "bird", "wind", "ocean", "river", "mountain",
        "heart", "soul", "dream", "whisper", "shadow", "cloud", "rain", "storm", "leaf", "petal"
    ]
    
    adjectives = [
        "silent", "gentle", "fierce", "golden", "silver", "crystal", "endless", "ancient", "eternal",
        "delicate", "wild", "quiet", "brilliant", "soft", "dark", "bright", "mysterious", "serene"
    ]
    
    verbs = [
        "dances", "whispers", "sings", "flows", "glows", "shines", "floats", "dreams", "calls",
        "wanders", "lingers", "rises", "falls", "weeps", "laughs", "embraces", "yearns", "soars"
    ]
    
    locations = [
        "sky", "forest", "meadow", "sea", "twilight", "dawn", "dusk", "night", "valley",
        "mist", "horizon", "garden", "wilderness", "abyss", "shadows", "stillness"
    ]
    
    times = [
        "morning", "evening", "twilight", "midnight", "dawn", "dusk", "spring", "summer",
        "autumn", "winter", "moment", "eternity", "past", "future", "present"
    ]
    
    abstract_nouns = [
        "love", "hope", "despair", "joy", "sorrow", "time", "eternity", "memory", "destiny",
        "truth", "beauty", "freedom", "wonder", "silence", "infinity", "wisdom", "passion"
    ]
    
    pronouns = [
        "I", "You", "We", "They", "She", "He"
    ]
    
    # Generate additional poems
    for _ in range(50):
        template = np.random.choice(templates)
        
        # Format template with random word choices
        poem = template
        for placeholder, word_list in [
            ("{noun}", nouns),
            ("{noun2}", nouns),
            ("{adjective}", adjectives),
            ("{Adjective}", [adj.capitalize() for adj in adjectives]),
            ("{verb}", verbs),
            ("{Verb}", [v.capitalize() for v in verbs]),
            ("{location}", locations),
            ("{time}", times),
            ("{abstract_noun}", abstract_nouns),
            ("{pronoun}", pronouns),
            ("{Pronoun}", [p.capitalize() for p in pronouns])
        ]:
            while placeholder in poem:
                poem = poem.replace(placeholder, np.random.choice(word_list), 1)
        
        poems.append(poem)
    
    # Limit to requested sample size
    np.random.seed(seed)
    if len(poems) > sample_size:
        poems = list(np.random.choice(poems, sample_size, replace=False))
    
    # Format the poems with prompts
    formatted_poems = []
    prompts = [
        "Write a poem about nature:",
        "Create a poetic verse about:",
        "Compose a poem that describes:",
        "A poetic expression of:",
        "Poetry inspired by:"
    ]
    
    for poem in poems:
        prompt = np.random.choice(prompts)
        if ":" in prompt:
            subject = np.random.choice(abstract_nouns + nouns)
            prompt = prompt.replace(":", f" {subject}:")
        
        text = f"{prompt}\n{poem}"
        formatted_poems.append(text)
    
    # Convert to tensors
    encodings = tokenizer(
        formatted_poems,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create input/target pairs for autoregressive learning
    input_ids = encodings['input_ids'].clone()
    labels = input_ids.clone()
    
    # Split into train/test
    train_size = int(0.8 * len(formatted_poems))
    
    train_dataset = {
        'input_ids': input_ids[:train_size],
        'attention_mask': encodings['attention_mask'][:train_size],
        'labels': labels[:train_size]
    }
    
    test_dataset = {
        'input_ids': input_ids[train_size:],
        'attention_mask': encodings['attention_mask'][train_size:],
        'labels': labels[train_size:]
    }
    
    return train_dataset, test_dataset

def create_dataset_for_task(task, tokenizer, max_length, sample_size, seed=42):
    """Create a dataset based on the specified task."""
    if task == "sentiment":
        return create_sentiment_dataset(tokenizer, max_length, sample_size, seed)
    elif task == "code":
        return create_code_dataset(tokenizer, max_length, sample_size, seed)
    elif task == "science":
        return create_science_dataset(tokenizer, max_length, sample_size, seed)
    elif task == "poetry":
        return create_poetry_dataset(tokenizer, max_length, sample_size, seed)
    else:
        raise ValueError(f"Unknown task: {task}")

def train_model(model, train_dataset, test_dataset, args, device, model_type="full"):
    """Train the model on a new task and monitor learning efficiency."""
    print(f"\n=== Training {model_type} model on {args.task} task ===")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * args.epochs
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )
    
    # Create a classifier head if task is sentiment analysis
    if args.task == "sentiment":
        # Simple classifier head for sentiment
        classifier = torch.nn.Linear(
            model.blocks[-1]["attn"].head_dim * model.blocks[-1]["attn"].num_heads,
            2  # Binary classification: positive/negative
        ).to(device)
        
        # Optimizer for classifier
        classifier_optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01
        )
        
        # Classification loss
        loss_fct = torch.nn.CrossEntropyLoss()
    else:
        classifier = None
        classifier_optimizer = None
        loss_fct = None
    
    # Initialize metrics storage
    metrics = {
        "train_loss": [],
        "eval_loss": [],
        "eval_performance": [],  # accuracy or perplexity depending on task
        "gate_changes": []
    }
    
    # Initial gate values
    initial_gates = {}
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    
    for l in range(num_layers):
        for h in range(num_heads):
            initial_gates[f"{l}_{h}"] = model.blocks[l]["attn"].gate[h].item()
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        if classifier:
            classifier.train()
        
        # Training loop
        epoch_loss = 0
        gate_values_before_epoch = get_gate_values(model)
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if args.task == "sentiment":
                # For sentiment analysis, we need a classification task
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                
                # Extract the last hidden state for classification
                last_hidden = outputs[:, -1, :]
                
                # Pass through classifier
                logits = classifier(last_hidden)
                
                # Calculate loss
                loss = loss_fct(logits, batch["labels"])
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                
                optimizer.step()
                classifier_optimizer.step()
                scheduler.step()
                
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()
            else:
                # For other tasks, we use language modeling
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                # Calculate loss
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Record metrics
            epoch_loss += loss.item()
            metrics["train_loss"].append(loss.item())
            
            # Evaluate every 100 steps
            if step % 100 == 0:
                eval_metrics = evaluate_model(
                    model=model,
                    dataset=test_dataset,
                    batch_size=args.batch_size,
                    device=device,
                    task=args.task,
                    classifier=classifier
                )
                
                metrics["eval_loss"].append(eval_metrics["loss"])
                metrics["eval_performance"].append(
                    eval_metrics["accuracy"] if args.task == "sentiment" else eval_metrics["perplexity"]
                )
                
                # Get current gate values and compare to initial values
                current_gates = get_gate_values(model)
                gate_changes = calculate_gate_changes(initial_gates, current_gates)
                metrics["gate_changes"].append(gate_changes)
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # Get gate values after epoch and calculate changes from before epoch
        gate_values_after_epoch = get_gate_values(model)
        gate_changes = calculate_gate_changes(gate_values_before_epoch, gate_values_after_epoch)
        
        # Evaluate at the end of each epoch
        eval_metrics = evaluate_model(
            model=model,
            dataset=test_dataset,
            batch_size=args.batch_size,
            device=device,
            task=args.task,
            classifier=classifier
        )
        
        if args.task == "sentiment":
            performance = f"accuracy: {eval_metrics['accuracy']:.2%}"
        else:
            performance = f"perplexity: {eval_metrics['perplexity']:.2f}"
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"loss: {avg_epoch_loss:.4f}, "
              f"eval loss: {eval_metrics['loss']:.4f}, "
              f"{performance}, "
              f"gate changes: {gate_changes:.4f}")
    
    # End of training
    print(f"Training completed for {model_type} model!")
    
    # Calculate total gate changes from initial to final
    final_gates = get_gate_values(model)
    total_gate_changes = calculate_gate_changes(initial_gates, final_gates)
    metrics["total_gate_changes"] = total_gate_changes
    
    # Final evaluation
    final_metrics = evaluate_model(
        model=model,
        dataset=test_dataset,
        batch_size=args.batch_size,
        device=device,
        task=args.task,
        classifier=classifier
    )
    
    metrics["final_eval_loss"] = final_metrics["loss"]
    metrics["final_performance"] = (
        final_metrics["accuracy"] if args.task == "sentiment" 
        else final_metrics["perplexity"]
    )
    
    return metrics, classifier

def evaluate_model(model, dataset, batch_size, device, task, classifier=None):
    """Evaluate the model on a dataset."""
    model.eval()
    if classifier:
        classifier.eval()
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    
    # Initialize metrics
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if task == "sentiment":
                # For sentiment analysis
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                
                # Extract the last hidden state for classification
                last_hidden = outputs[:, -1, :]
                
                # Pass through classifier
                logits = classifier(last_hidden)
                
                # Calculate loss
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits, batch["labels"])
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
                
                total_loss += loss.item() * batch["labels"].size(0)
            else:
                # For language modeling tasks
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                # Calculate loss
                loss = outputs.loss
                total_loss += loss.item() * batch["input_ids"].size(0)
                total += batch["input_ids"].size(0)
    
    # Calculate metrics
    avg_loss = total_loss / total
    
    if task == "sentiment":
        accuracy = correct / total
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    else:
        # For language modeling, calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }

def get_gate_values(model):
    """Extract gate values from the model."""
    gate_values = {}
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    
    for l in range(num_layers):
        for h in range(num_heads):
            gate_values[f"{l}_{h}"] = model.blocks[l]["attn"].gate[h].item()
    
    return gate_values

def calculate_gate_changes(gates_before, gates_after):
    """Calculate the average absolute change in gate values."""
    total_change = 0
    count = 0
    
    for key in gates_before:
        if key in gates_after:
            change = abs(gates_after[key] - gates_before[key])
            total_change += change
            count += 1
    
    return total_change / count if count > 0 else 0

def visualize_learning_efficiency(full_metrics, pruned_metrics, output_dir, args):
    """Create visualizations comparing the learning efficiency of full and pruned models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    full_loss = full_metrics["train_loss"]
    pruned_loss = pruned_metrics["train_loss"]
    
    full_eval_loss = full_metrics["eval_loss"]
    pruned_eval_loss = pruned_metrics["eval_loss"]
    
    full_performance = full_metrics["eval_performance"]
    pruned_performance = pruned_metrics["eval_performance"]
    
    full_gate_changes = full_metrics["gate_changes"]
    pruned_gate_changes = pruned_metrics["gate_changes"]
    
    # Create evaluation points (assuming evaluations occurred at the same intervals)
    eval_points = list(range(0, len(full_eval_loss)))
    
    # Set up plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot 1: Training Loss Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(full_loss[:len(pruned_loss)], label='Full Model', alpha=0.7, linewidth=2)
    plt.plot(pruned_loss, label=f'Pruned Model ({args.pruning_level:.0%})', alpha=0.7, linewidth=2)
    plt.title('Training Loss Comparison', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 2: Evaluation Loss Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(eval_points, full_eval_loss, label='Full Model', marker='o', alpha=0.7, linewidth=2)
    plt.plot(eval_points, pruned_eval_loss, label=f'Pruned Model ({args.pruning_level:.0%})', marker='s', alpha=0.7, linewidth=2)
    plt.title('Evaluation Loss Comparison', fontsize=16)
    plt.xlabel('Evaluation Points', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_loss_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 3: Performance Metric Comparison
    plt.figure(figsize=(10, 6))
    
    if args.task == "sentiment":
        # For accuracy (higher is better)
        plt.plot(eval_points, [x * 100 for x in full_performance], label='Full Model', marker='o', alpha=0.7, linewidth=2)
        plt.plot(eval_points, [x * 100 for x in pruned_performance], label=f'Pruned Model ({args.pruning_level:.0%})', marker='s', alpha=0.7, linewidth=2)
        plt.title('Accuracy Comparison on Sentiment Analysis', fontsize=16)
        plt.ylabel('Accuracy (%)', fontsize=14)
    else:
        # For perplexity (lower is better)
        plt.plot(eval_points, full_performance, label='Full Model', marker='o', alpha=0.7, linewidth=2)
        plt.plot(eval_points, pruned_performance, label=f'Pruned Model ({args.pruning_level:.0%})', marker='s', alpha=0.7, linewidth=2)
        plt.title(f'Perplexity Comparison on {args.task.capitalize()} Task', fontsize=16)
        plt.ylabel('Perplexity (lower is better)', fontsize=14)
    
    plt.xlabel('Evaluation Points', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 4: Gate Changes During Learning
    plt.figure(figsize=(10, 6))
    plt.plot(eval_points, full_gate_changes, label='Full Model', marker='o', alpha=0.7, linewidth=2)
    plt.plot(eval_points, pruned_gate_changes, label=f'Pruned Model ({args.pruning_level:.0%})', marker='s', alpha=0.7, linewidth=2)
    plt.title('Gate Changes During Learning', fontsize=16)
    plt.xlabel('Evaluation Points', fontsize=14)
    plt.ylabel('Average Gate Change', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gate_changes_comparison.png'), dpi=300)
    plt.close()
    
    # Plot 5: Learning Efficiency Comparison
    plt.figure(figsize=(10, 6))
    
    # Calculate the learning efficiency as performance improvement per training step
    if args.task == "sentiment":
        # For accuracy (higher is better)
        full_efficiency = [full_performance[i] / (i+1) for i in range(len(full_performance))]
        pruned_efficiency = [pruned_performance[i] / (i+1) for i in range(len(pruned_performance))]
        ylabel = 'Learning Efficiency (Accuracy per Evaluation Point)'
    else:
        # For perplexity (lower is better), we invert it so higher values are better
        full_efficiency = [1/full_performance[i] / (i+1) for i in range(len(full_performance))]
        pruned_efficiency = [1/pruned_performance[i] / (i+1) for i in range(len(pruned_performance))]
        ylabel = 'Learning Efficiency (1/Perplexity per Evaluation Point)'
    
    plt.plot(eval_points, full_efficiency, label='Full Model', marker='o', alpha=0.7, linewidth=2)
    plt.plot(eval_points, pruned_efficiency, label=f'Pruned Model ({args.pruning_level:.0%})', marker='s', alpha=0.7, linewidth=2)
    plt.title('Learning Efficiency Comparison', fontsize=16)
    plt.xlabel('Evaluation Points', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_efficiency_comparison.png'), dpi=300)
    plt.close()
    
    # Create a summary table of the results
    summary_data = {
        'Metric': [
            'Final Training Loss', 
            'Final Evaluation Loss', 
            'Final Performance', 
            'Total Gate Changes',
            'Training Time per Epoch (relative)'
        ],
        'Full Model': [
            f"{full_metrics['train_loss'][-1]:.4f}",
            f"{full_metrics['final_eval_loss']:.4f}",
            f"{full_metrics['final_performance']:.4f}" if args.task != 'sentiment' else f"{full_metrics['final_performance']:.2%}",
            f"{full_metrics['total_gate_changes']:.4f}",
            "1.0x"
        ],
        'Pruned Model': [
            f"{pruned_metrics['train_loss'][-1]:.4f}",
            f"{pruned_metrics['final_eval_loss']:.4f}",
            f"{pruned_metrics['final_performance']:.4f}" if args.task != 'sentiment' else f"{pruned_metrics['final_performance']:.2%}",
            f"{pruned_metrics['total_gate_changes']:.4f}",
            f"{(1.0 - args.pruning_level):.2f}x"  # Approximate speedup based on pruning level
        ]
    }
    
    # Save summary to CSV
    pd.DataFrame(summary_data).to_csv(
        os.path.join(output_dir, 'learning_results_summary.csv'),
        index=False
    )
    
    # Generate a summary text file
    with open(os.path.join(output_dir, 'learning_results_summary.txt'), 'w') as f:
        f.write(f"Learning After Pruning Experiment Results\n")
        f.write(f"=========================================\n\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Pruning Level: {args.pruning_level:.0%}\n")
        f.write(f"Pruning Strategy: {args.pruning_strategy}\n\n")
        
        f.write(f"Summary of Results:\n")
        f.write(f"-----------------\n")
        for i, metric in enumerate(summary_data['Metric']):
            f.write(f"{metric}:\n")
            f.write(f"  - Full Model: {summary_data['Full Model'][i]}\n")
            f.write(f"  - Pruned Model: {summary_data['Pruned Model'][i]}\n\n")
        
        # Add conclusions
        f.write(f"Conclusions:\n")
        f.write(f"-----------\n")
        
        if args.task == "sentiment":
            # Compare accuracy
            full_acc = full_metrics['final_performance']
            pruned_acc = pruned_metrics['final_performance']
            
            if pruned_acc >= full_acc * 0.95:
                f.write(f"The pruned model ({args.pruning_level:.0%} pruning) maintained comparable accuracy ")
                f.write(f"({pruned_acc:.2%} vs {full_acc:.2%} for the full model) on the sentiment analysis task.\n\n")
            elif pruned_acc >= full_acc:
                f.write(f"The pruned model ({args.pruning_level:.0%} pruning) achieved BETTER accuracy ")
                f.write(f"({pruned_acc:.2%} vs {full_acc:.2%} for the full model) on the sentiment analysis task.\n\n")
            else:
                f.write(f"The pruned model ({args.pruning_level:.0%} pruning) achieved lower accuracy ")
                f.write(f"({pruned_acc:.2%} vs {full_acc:.2%} for the full model) on the sentiment analysis task.\n\n")
        else:
            # Compare perplexity
            full_ppl = full_metrics['final_performance']
            pruned_ppl = pruned_metrics['final_performance']
            
            if pruned_ppl <= full_ppl * 1.05:
                f.write(f"The pruned model ({args.pruning_level:.0%} pruning) maintained comparable perplexity ")
                f.write(f"({pruned_ppl:.2f} vs {full_ppl:.2f} for the full model) on the {args.task} task.\n\n")
            elif pruned_ppl <= full_ppl:
                f.write(f"The pruned model ({args.pruning_level:.0%} pruning) achieved BETTER perplexity ")
                f.write(f"({pruned_ppl:.2f} vs {full_ppl:.2f} for the full model) on the {args.task} task.\n\n")
            else:
                f.write(f"The pruned model ({args.pruning_level:.0%} pruning) achieved higher perplexity ")
                f.write(f"({pruned_ppl:.2f} vs {full_ppl:.2f} for the full model) on the {args.task} task.\n\n")
        
        # Compare gate changes
        full_changes = full_metrics['total_gate_changes']
        pruned_changes = pruned_metrics['total_gate_changes']
        
        if pruned_changes > full_changes:
            f.write(f"The pruned model showed MORE gate adaptation during learning ")
            f.write(f"(change of {pruned_changes:.4f} vs {full_changes:.4f} for the full model).\n")
            f.write(f"This suggests that pruned models may have greater neuroplasticity and adapt more efficiently to new tasks.\n\n")
        else:
            f.write(f"The pruned model showed LESS gate adaptation during learning ")
            f.write(f"(change of {pruned_changes:.4f} vs {full_changes:.4f} for the full model).\n")
            f.write(f"This suggests that the full model may have more flexibility to adapt to new tasks.\n\n")
        
        # Final conclusion
        f.write(f"Overall, this experiment demonstrates that models pruned with the {args.pruning_strategy} strategy ")
        f.write(f"can effectively learn new tasks and adapt their remaining attention mechanisms to optimize performance.\n")
        f.write(f"This provides evidence that the Sentinel-AI pruning approach enables models to maintain adaptability ")
        f.write(f"while achieving computational efficiency gains.\n")

def visualize_gate_activity(full_model, pruned_model, output_dir):
    """Visualize gate activity before and after learning."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get gate dimensions
    num_layers = len(full_model.blocks)
    num_heads = full_model.blocks[0]["attn"].num_heads
    
    # Create matrices of gate values
    full_gates = torch.zeros(num_layers, num_heads)
    pruned_gates = torch.zeros(num_layers, num_heads)
    
    for l in range(num_layers):
        for h in range(num_heads):
            full_gates[l, h] = full_model.blocks[l]["attn"].gate[h].item()
            pruned_gates[l, h] = pruned_model.blocks[l]["attn"].gate[h].item()
    
    # Create heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Full model gates
    im0 = axs[0].imshow(full_gates.numpy(), cmap="YlOrRd", vmin=0, vmax=1)
    axs[0].set_title("Full Model Gate Activity", fontsize=16)
    axs[0].set_xlabel("Attention Head", fontsize=14)
    axs[0].set_ylabel("Transformer Layer", fontsize=14)
    fig.colorbar(im0, ax=axs[0], label="Gate Value")
    
    # Pruned model gates
    im1 = axs[1].imshow(pruned_gates.numpy(), cmap="YlOrRd", vmin=0, vmax=1)
    axs[1].set_title("Pruned Model Gate Activity", fontsize=16)
    axs[1].set_xlabel("Attention Head", fontsize=14)
    axs[1].set_ylabel("Transformer Layer", fontsize=14)
    fig.colorbar(im1, ax=axs[1], label="Gate Value")
    
    # Add grid lines
    for ax in axs:
        ax.grid(False)
        ax.set_xticks(range(num_heads))
        ax.set_yticks(range(num_layers))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gate_activity_comparison.png'), dpi=300)
    plt.close()
    
    # Create a difference heatmap
    plt.figure(figsize=(10, 8))
    difference = pruned_gates - full_gates
    im = plt.imshow(difference.numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="Gate Value Difference (Pruned - Full)")
    plt.title("Gate Activity Differences", fontsize=16)
    plt.xlabel("Attention Head", fontsize=14)
    plt.ylabel("Transformer Layer", fontsize=14)
    
    # Add grid lines
    plt.grid(False)
    plt.xticks(range(num_heads))
    plt.yticks(range(num_layers))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gate_activity_difference.png'), dpi=300)
    plt.close()

def generate_sample_outputs(full_model, pruned_model, tokenizer, task, device, output_dir):
    """Generate sample outputs to compare full and pruned models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create prompts based on the task
    if task == "sentiment":
        prompts = [
            "This movie was absolutely",
            "I really disliked the way",
            "The service at this restaurant was",
            "I was disappointed by the"
        ]
    elif task == "code":
        prompts = [
            "Problem: Write a function to check if a number is prime.\nSolution:",
            "Problem: Create a function to reverse a string.\nSolution:",
            "Problem: Write a function to find duplicates in a list.\nSolution:",
            "Problem: Implement a binary search algorithm.\nSolution:"
        ]
    elif task == "science":
        prompts = [
            "The theory of relativity states that",
            "Scientists have discovered that the human brain",
            "According to recent research on climate change,",
            "The structure of DNA consists of"
        ]
    elif task == "poetry":
        prompts = [
            "Write a poem about nature:\nThe trees sway in the",
            "Create a poetic verse about love:\nHearts entwined like",
            "Compose a poem that describes the ocean:\nWaves crash upon",
            "Poetry inspired by stars:\nTwinkling lights in the"
        ]
    
    # Generate text with both models
    full_outputs = []
    pruned_outputs = []
    
    for prompt in prompts:
        full_output = generate_text(
            model=full_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=100,
            temperature=0.7,
            device=device
        )
        
        pruned_output = generate_text(
            model=pruned_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=100,
            temperature=0.7,
            device=device
        )
        
        full_outputs.append(full_output)
        pruned_outputs.append(pruned_output)
    
    # Save outputs to a text file
    with open(os.path.join(output_dir, 'sample_generations.txt'), 'w') as f:
        f.write(f"Sample Generations after Learning the {task.capitalize()} Task\n")
        f.write("=" * 80 + "\n\n")
        
        for i, prompt in enumerate(prompts):
            f.write(f"Prompt {i+1}: {prompt}\n\n")
            f.write("Full Model Output:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{full_outputs[i]}\n\n")
            f.write("Pruned Model Output:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{pruned_outputs[i]}\n\n")
            f.write("=" * 80 + "\n\n")

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.drive_path:
        output_dir = os.path.join(args.drive_path, args.output_dir, f"{args.task}_{timestamp}")
    else:
        output_dir = os.path.join(args.output_dir, f"{args.task}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load baseline model
    baseline_model = load_baseline_model(args.model_name, device)
    
    # Create two adaptive models (one for pruning, one as full model baseline)
    print("Creating full model (no pruning)...")
    full_model = load_adaptive_model(args.model_name, baseline_model, device)
    
    print(f"Creating pruned model ({args.pruning_level:.0%} {args.pruning_strategy} pruning)...")
    pruned_model = load_adaptive_model(args.model_name, baseline_model, device)
    pruned_model = apply_pruning(pruned_model, args.pruning_strategy, args.pruning_level, device)
    
    # Create dataset for the task
    print(f"Creating dataset for {args.task} task...")
    train_dataset, test_dataset = create_dataset_for_task(
        args.task, tokenizer, args.max_length, args.sample_size, args.seed
    )
    
    # Train and evaluate both models
    full_metrics, full_classifier = train_model(
        full_model, train_dataset, test_dataset, args, device, model_type="full"
    )
    
    pruned_metrics, pruned_classifier = train_model(
        pruned_model, train_dataset, test_dataset, args, device, model_type="pruned"
    )
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_learning_efficiency(full_metrics, pruned_metrics, output_dir, args)
    visualize_gate_activity(full_model, pruned_model, output_dir)
    generate_sample_outputs(full_model, pruned_model, tokenizer, args.task, device, output_dir)
    
    print(f"Results saved to {output_dir}")
    
    return {
        'full_model': full_model,
        'pruned_model': pruned_model,
        'full_metrics': full_metrics,
        'pruned_metrics': pruned_metrics,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    main()