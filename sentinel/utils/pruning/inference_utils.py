"""
Inference utilities for pruned models.

This module provides functions for text generation, degeneration detection,
and other inference-related utilities for pruned transformer models.
"""


def check_for_degeneration(text, repetition_threshold=3, diversity_threshold=0.4):
    """
    Check if generated text shows signs of degeneration.
    
    Args:
        text: Generated text
        repetition_threshold: Max acceptable repetitions
        diversity_threshold: Min acceptable diversity
        
    Returns:
        Tuple of (degeneration_detected, degeneration_score, info_dict)
    """
    # Placeholder implementation
    import random
    
    # For placeholder, use simple heuristics and random values
    words = text.split()
    
    # Count repeated words
    word_counts = {}
    for word in words:
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    
    max_repetition = max(word_counts.values()) if word_counts else 0
    unique_ratio = len(word_counts) / max(1, len(words))
    
    # Add some randomness for placeholder
    degeneration_score = 0.5 * max(0, (max_repetition - repetition_threshold) / 5) + \
                         0.5 * max(0, (diversity_threshold - unique_ratio) / 0.4) + \
                         random.uniform(-0.2, 0.2)  # Random jitter
    
    degeneration_score = max(0.0, min(5.0, degeneration_score))
    degeneration_detected = degeneration_score > 1.0
    
    info_dict = {
        "max_repetition": max_repetition,
        "unique_ratio": unique_ratio,
        "repetitions": {word: count for word, count in word_counts.items() if count > repetition_threshold}
    }
    
    return degeneration_detected, degeneration_score, info_dict


def apply_degeneration_penalty(perplexity, degeneration_score, max_penalty=1.0):
    """
    Apply a perplexity penalty based on degeneration score.
    
    Args:
        perplexity: Base perplexity
        degeneration_score: How much degeneration was detected
        max_penalty: Maximum penalty factor to apply
        
    Returns:
        Adjusted perplexity
    """
    # Placeholder implementation
    penalty = min(max_penalty, degeneration_score * 0.2)
    return perplexity * (1.0 + penalty)


def display_generation(prompt, generation, max_chars=100):
    """
    Display a text generation in a formatted way.
    
    Args:
        prompt: Input prompt
        generation: Generated text
        max_chars: Maximum characters to display
    """
    print(f"Prompt: {prompt}")
    if len(generation) > max_chars:
        print(f"Generation: {generation[:max_chars]}...")
    else:
        print(f"Generation: {generation}")
    print("-" * 50)


def display_side_by_side(prompt, text1, text2, max_chars=80):
    """
    Display two text generations side by side for comparison.
    
    Args:
        prompt: Input prompt
        text1: First text (e.g., baseline model)
        text2: Second text (e.g., pruned model)
        max_chars: Maximum characters per line
    """
    print(f"Prompt: {prompt}")
    print("-" * 100)
    print(f"{'Baseline Model':<50} | {'Pruned Model':<50}")
    print("-" * 100)
    
    # Get lines for each text
    lines1 = get_wrapped_lines(text1, max_chars)
    lines2 = get_wrapped_lines(text2, max_chars)
    
    # Ensure same number of lines
    while len(lines1) < len(lines2):
        lines1.append("")
    while len(lines2) < len(lines1):
        lines2.append("")
    
    # Display side by side
    for i in range(len(lines1)):
        print(f"{lines1[i]:<50} | {lines2[i]:<50}")
    
    print("-" * 100)


def get_wrapped_lines(text, max_chars):
    """
    Split text into lines of maximum length.
    
    Args:
        text: Text to wrap
        max_chars: Maximum characters per line
        
    Returns:
        List of lines
    """
    lines = []
    current_line = ""
    
    for word in text.split():
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += (" " + word if current_line else word)
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines


def get_test_prompts(category="general", length="short"):
    """
    Get a list of test prompts for evaluating generation.
    
    Args:
        category: Type of prompt ("general", "technical", "transformer")
        length: Length of prompt ("short", "medium", "long")
        
    Returns:
        List of prompts
    """
    prompts = {
        "general": {
            "short": [
                "Once upon a time",
                "The future of AI",
                "In the beginning",
                "The most important thing",
                "Renewable energy sources"
            ],
            "medium": [
                "Once upon a time, in a land far away, there lived a young inventor",
                "The future of AI depends on several key developments in the coming years",
                "In the beginning, the universe was created. This has made a lot of people very angry",
                "The most important thing to remember when designing a system is to keep it simple",
                "Renewable energy sources offer a sustainable alternative to fossil fuels"
            ]
        },
        "technical": {
            "short": [
                "The algorithm works by",
                "Neural networks can",
                "When training a model",
                "A key limitation of",
                "The architecture consists of"
            ],
            "medium": [
                "The algorithm works by iteratively updating the weights based on gradient information",
                "Neural networks can approximate any continuous function given enough parameters",
                "When training a model with limited data, regularization becomes extremely important",
                "A key limitation of current language models is their inability to reason about",
                "The architecture consists of multiple attention layers followed by feed-forward networks"
            ]
        },
        "transformer": {
            "short": [
                "Attention mechanisms allow",
                "Transformer models process",
                "Language generation requires",
                "The self-attention layer",
                "Transformer architectures excel at"
            ],
            "medium": [
                "Attention mechanisms allow models to focus on different parts of the input sequence",
                "Transformer models process all tokens in parallel, unlike recurrent neural networks",
                "Language generation requires careful handling of context and coherence constraints",
                "The self-attention layer computes relationships between all pairs of positions",
                "Transformer architectures excel at capturing long-range dependencies in sequences"
            ]
        }
    }
    
    # Return the requested prompts
    if category in prompts and length in prompts[category]:
        return prompts[category][length]
    else:
        # Return default prompts if the requested combination isn't available
        return prompts["general"]["short"]