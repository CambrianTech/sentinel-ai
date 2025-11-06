# sentinel_data/eval.py

# Import directly from the local dataset_loader
from sentinel_data.dataset_loader import load_dataset

def load_eval_prompts(dataset_name=None, num_prompts=5):
    """Load evaluation prompts for the specified dataset"""
    # Simple prompts for evaluation
    default_prompts = [
        "The transformer model processes data through multiple layers of computation.",
        "Artificial intelligence systems can learn from experience and improve over time.",
        "The neural network was trained to recognize patterns in complex datasets.",
        "Language models predict the next token based on previous context.",
        "The attention mechanism allows the model to focus on relevant parts of the input."
    ]
    
    return default_prompts[:num_prompts]