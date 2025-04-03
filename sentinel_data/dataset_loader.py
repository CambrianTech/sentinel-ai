# sentinel_data/dataset_loader.py

import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class SimpleTokenizedDataset(Dataset):
    """Simple dataset for language modeling with fixed text."""
    
    def __init__(self, text, tokenizer, max_length=128):
        """
        Initialize a tokenized dataset with a simple fixed text.
        
        Args:
            text: Text to tokenize
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize entire text
        self.tokens = self.tokenizer.encode(text)
        
        # Create chunks of max_length
        self.chunks = []
        for i in range(0, len(self.tokens) - max_length, max_length // 2):
            self.chunks.append(self.tokens[i:i + max_length])
        
        if len(self.chunks) == 0 and len(self.tokens) > 0:
            # If text is too short, pad it
            chunk = self.tokens + [tokenizer.pad_token_id] * (max_length - len(self.tokens))
            self.chunks = [chunk]
        
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        # Get sequence of tokens
        tokens = self.chunks[idx]
        attention_mask = [1] * len(tokens)
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        # Create input_ids and labels (shifted for next token prediction)
        input_ids = torch.tensor(tokens)
        attention_mask = torch.tensor(attention_mask)
        
        # For causal language modeling, labels are the same as inputs
        # (shifted by 1 during loss calculation)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def load_dataset(dataset_name, tokenizer, max_length=None):
    """
    Load and prepare datasets for training and evaluation.
    
    Args:
        dataset_name: Name of the dataset to load
        tokenizer: Tokenizer to use for tokenization
        max_length: Maximum sequence length (None for no limit)
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Simple Shakespeare text for quick experiments
    shakespeare_text = """
    ROMEO: But, soft! what light through yonder window breaks?
    It is the east, and Juliet is the sun.
    Arise, fair sun, and kill the envious moon,
    Who is already sick and pale with grief,
    That thou her maid art far more fair than she.

    JULIET: O Romeo, Romeo! wherefore art thou Romeo?
    Deny thy father and refuse thy name;
    Or, if thou wilt not, be but sworn my love,
    And I'll no longer be a Capulet.

    ROMEO: Shall I hear more, or shall I speak at this?

    JULIET: 'Tis but thy name that is my enemy;
    Thou art thyself, though not a Montague.
    What's Montague? it is nor hand, nor foot,
    Nor arm, nor face, nor any other part
    Belonging to a man. O, be some other name!
    What's in a name? that which we call a rose
    By any other name would smell as sweet;
    So Romeo would, were he not Romeo call'd,
    Retain that dear perfection which he owes
    Without that title. Romeo, doff thy name,
    And for that name which is no part of thee
    Take all myself.
    """
    
    # Set appropriate max length
    if max_length is None:
        max_length = 128
    
    # Create datasets
    full_dataset = SimpleTokenizedDataset(shakespeare_text, tokenizer, max_length)
    
    # Split into train and eval (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, eval_size]
    )
    
    # Wrap datasets with DataLoader
    class DatasetWrapper:
        def __init__(self, train_dataset, eval_dataset, batch_size=8):
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.batch_size = batch_size
            self.tokenizer = tokenizer
            
            # Create iterators
            self._train_iterator = self._create_train_iterator()
            self._eval_iterator = self._create_eval_iterator()
        
        def set_tokenizer(self, new_tokenizer):
            self.tokenizer = new_tokenizer
        
        def _create_train_iterator(self):
            train_loader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size,
                shuffle=True
            )
            
            # Convert to JAX format
            while True:
                for batch in train_loader:
                    # Convert to JAX format
                    jax_batch = {
                        "input_ids": batch["input_ids"].numpy(),
                        "attention_mask": batch["attention_mask"].numpy()
                    }
                    yield jax_batch
        
        def _create_eval_iterator(self):
            eval_loader = DataLoader(
                self.eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False
            )
            
            # Convert to JAX format
            while True:
                for batch in eval_loader:
                    # Convert to JAX format
                    jax_batch = {
                        "input_ids": batch["input_ids"].numpy(),
                        "attention_mask": batch["attention_mask"].numpy()
                    }
                    yield jax_batch
        
        @property
        def train_dataloader(self):
            return self._train_iterator
        
        @property
        def val_dataloader(self):
            return self._eval_iterator
        
        def get_evaluation_samples(self, num_samples=5):
            """Get sample texts for evaluation"""
            # Generate prompts for evaluation
            prompts = [
                "Romeo: ",
                "Juliet: ",
                "What's in a name? ",
                "It is the east, and ",
                "Arise, fair sun, and "
            ]
            return prompts[:num_samples]
    
    # Create and return the wrapper
    return DatasetWrapper(train_dataset, eval_dataset, batch_size=8)