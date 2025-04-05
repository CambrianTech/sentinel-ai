#!/usr/bin/env python
"""
Task Suite for Neural Plasticity Stress Testing

This module provides specialized task suites for stress testing neural plasticity,
including diverse tasks like commonsense QA, summarization, and code completion.
The module enables testing how models adapt to different tasks and how function
is preserved across task switches.
"""

import os
import torch
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field

from transformers import PreTrainedTokenizer, PreTrainedModel

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TaskExample:
    """A single task example with input, expected output, and metadata"""
    
    input_text: str
    expected_output: Optional[str] = None
    task_type: str = "generation"  # generation, classification, qa, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for a specific task"""
    
    name: str
    description: str
    metric: str  # e.g., "accuracy", "f1", "rouge", "bleu", "perplexity"
    examples: List[TaskExample] = field(default_factory=list)
    max_input_length: int = 512
    max_output_length: int = 128
    task_type: str = "generation"  # generation, classification, qa, etc.


class TaskSuite:
    """
    Collection of tasks for stress testing neural plasticity.
    
    This class provides a standardized interface for defining, loading, and
    evaluating tasks for stress testing neural plasticity. It includes methods
    for creating task-specific dataloaders and evaluation routines.
    """
    
    def __init__(
        self,
        name: str,
        tasks: List[TaskConfig],
        device: Optional[str] = None
    ):
        """
        Initialize a task suite.
        
        Args:
            name: Name of the task suite
            tasks: List of task configurations
            device: Device to run computations on
        """
        self.name = name
        self.tasks = {task.name: task for task in tasks}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized TaskSuite '{name}' with {len(tasks)} tasks")
    
    def get_task(self, task_name: str) -> Optional[TaskConfig]:
        """Get a task by name"""
        return self.tasks.get(task_name)
    
    def get_task_names(self) -> List[str]:
        """Get list of all task names"""
        return list(self.tasks.keys())
    
    def create_dataloader(
        self,
        task_name: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """
        Create a dataloader for a specific task.
        
        Args:
            task_name: Name of the task
            tokenizer: Tokenizer for the model
            batch_size: Batch size for the dataloader
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the task
        """
        task = self.get_task(task_name)
        if task is None:
            raise ValueError(f"Unknown task: {task_name}")
        
        # Create dataset
        inputs = []
        outputs = []
        
        for example in task.examples:
            inputs.append(example.input_text)
            outputs.append(example.expected_output if example.expected_output else "")
        
        from torch.utils.data import Dataset, DataLoader
        
        class TaskDataset(Dataset):
            def __init__(self, inputs, outputs, tokenizer, task_type, max_input_length, max_output_length):
                self.inputs = inputs
                self.outputs = outputs
                self.tokenizer = tokenizer
                self.task_type = task_type
                self.max_input_length = max_input_length
                self.max_output_length = max_output_length
            
            def __len__(self):
                return len(self.inputs)
                
            def __getitem__(self, idx):
                input_text = self.inputs[idx]
                output_text = self.outputs[idx]
                
                # For language modeling tasks
                if self.task_type == "generation" or self.task_type == "lm":
                    # Encode input
                    input_encoding = self.tokenizer(
                        input_text,
                        truncation=True,
                        max_length=self.max_input_length,
                        return_tensors="pt"
                    )
                    
                    # Create combined input-output for labels
                    combined_text = input_text + output_text
                    combined_encoding = self.tokenizer(
                        combined_text,
                        truncation=True,
                        max_length=self.max_input_length + self.max_output_length,
                        return_tensors="pt"
                    )
                    
                    # Create labels with -100 for input tokens
                    labels = combined_encoding["input_ids"].clone()
                    input_length = input_encoding["input_ids"].shape[1]
                    labels[0, :input_length] = -100  # Mask input tokens
                    
                    return {
                        "input_ids": combined_encoding["input_ids"][0],
                        "attention_mask": combined_encoding["attention_mask"][0],
                        "labels": labels[0]
                    }
                
                # For classification tasks
                elif self.task_type == "classification":
                    # Convert output to class index
                    try:
                        label = int(output_text)
                    except ValueError:
                        # Try to find in label map
                        label_map = getattr(self, "label_map", {})
                        label = label_map.get(output_text, 0)
                    
                    # Encode input
                    encoding = self.tokenizer(
                        input_text,
                        truncation=True,
                        max_length=self.max_input_length,
                        return_tensors="pt"
                    )
                    
                    return {
                        "input_ids": encoding["input_ids"][0],
                        "attention_mask": encoding["attention_mask"][0],
                        "labels": torch.tensor(label)
                    }
                
                # For question answering
                elif self.task_type == "qa":
                    # Split input into context and question if needed
                    parts = input_text.split("[SEP]")
                    if len(parts) > 1:
                        context = parts[0].strip()
                        question = parts[1].strip()
                    else:
                        context = ""
                        question = input_text
                    
                    # Encode
                    encoding = self.tokenizer(
                        question,
                        context,
                        truncation="only_second",
                        max_length=self.max_input_length,
                        return_tensors="pt"
                    )
                    
                    # Add answer span if available
                    if output_text:
                        # Simple heuristic: find the start of the answer in the context
                        if output_text in context:
                            start_idx = context.find(output_text)
                            end_idx = start_idx + len(output_text)
                            
                            # Convert character indices to token indices (simple approximation)
                            # This is a simplification and might not be accurate for all tokenizers
                            tokens = self.tokenizer.encode(context)
                            token_start_idx = 0
                            token_end_idx = len(tokens) - 1
                            
                            encoding["start_positions"] = torch.tensor([token_start_idx])
                            encoding["end_positions"] = torch.tensor([token_end_idx])
                    
                    return {
                        "input_ids": encoding["input_ids"][0],
                        "attention_mask": encoding["attention_mask"][0],
                        "labels": self.tokenizer.encode(output_text, return_tensors="pt")[0] if output_text else None
                    }
                
                # Default to basic encoding
                else:
                    encoding = self.tokenizer(
                        input_text,
                        truncation=True,
                        max_length=self.max_input_length,
                        return_tensors="pt"
                    )
                    
                    return {
                        "input_ids": encoding["input_ids"][0],
                        "attention_mask": encoding["attention_mask"][0],
                        "labels": self.tokenizer.encode(output_text, return_tensors="pt")[0] if output_text else None
                    }
        
        dataset = TaskDataset(
            inputs,
            outputs,
            tokenizer,
            task.task_type,
            task.max_input_length,
            task.max_output_length
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        return dataloader
    
    def evaluate(
        self,
        task_name: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a specific task.
        
        Args:
            task_name: Name of the task
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to run evaluation on
            
        Returns:
            Dictionary with evaluation metrics
        """
        task = self.get_task(task_name)
        if task is None:
            raise ValueError(f"Unknown task: {task_name}")
        
        device = device or self.device
        model = model.to(device)
        model.eval()
        
        # Initialize metrics
        metrics = {
            "task": task_name,
            "type": task.task_type,
            "metric": task.metric,
            "score": 0.0
        }
        
        # Simple loop for toy tasks
        correct = 0
        total = 0
        
        for example in task.examples:
            # Encode input
            inputs = tokenizer(
                example.input_text,
                return_tensors="pt"
            ).to(device)
            
            # Generate output
            with torch.no_grad():
                if task.task_type == "generation" or task.task_type == "lm":
                    # For generation tasks
                    outputs = model.generate(
                        **inputs,
                        max_length=task.max_input_length + task.max_output_length,
                        num_return_sequences=1,
                        do_sample=False
                    )
                    
                    # Decode output
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Remove input prefix if possible
                    if generated_text.startswith(example.input_text):
                        generated_text = generated_text[len(example.input_text):].strip()
                    
                    # Compare with expected output
                    if example.expected_output:
                        # For simplicity, consider exact match
                        if generated_text.strip() == example.expected_output.strip():
                            correct += 1
                        total += 1
                
                elif task.task_type == "classification":
                    # For classification tasks
                    outputs = model(**inputs)
                    
                    # Get predicted class
                    predicted_class = outputs.logits.argmax(-1).item()
                    
                    # Compare with expected class
                    expected_class = int(example.expected_output) if example.expected_output else 0
                    if predicted_class == expected_class:
                        correct += 1
                    total += 1
                
                elif task.task_type == "qa":
                    # For QA tasks
                    outputs = model(**inputs)
                    
                    # Extract answer span
                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits
                    
                    start_idx = torch.argmax(start_logits).item()
                    end_idx = torch.argmax(end_logits).item()
                    
                    # Get answer tokens
                    answer_tokens = inputs.input_ids[0][start_idx:end_idx+1]
                    predicted_answer = tokenizer.decode(answer_tokens)
                    
                    # Compare with expected answer
                    if example.expected_output and predicted_answer.strip() == example.expected_output.strip():
                        correct += 1
                    total += 1
        
        # Calculate metrics
        if total > 0:
            if task.metric == "accuracy":
                metrics["score"] = correct / total
            elif task.metric == "perplexity":
                # For perplexity, we need to run through the dataloader
                dataloader = self.create_dataloader(task_name, tokenizer, batch_size=4)
                
                total_loss = 0
                total_tokens = 0
                
                with torch.no_grad():
                    for batch in dataloader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**batch)
                        loss = outputs.loss
                        
                        # Track loss and token count
                        total_loss += loss.item() * batch["input_ids"].size(0)
                        total_tokens += batch["input_ids"].size(0) * batch["input_ids"].size(1)
                
                # Calculate perplexity
                avg_loss = total_loss / total_tokens
                metrics["score"] = torch.exp(torch.tensor(avg_loss)).item()
            else:
                # Default to accuracy
                metrics["score"] = correct / total
        
        return metrics


def create_diverse_task_suite() -> TaskSuite:
    """
    Create a diverse task suite with commonsense QA, summarization, and code completion.
    
    Returns:
        TaskSuite with diverse tasks
    """
    # Commonsense QA task
    commonsense_qa = TaskConfig(
        name="commonsense_qa",
        description="Simple commonsense questions",
        metric="accuracy",
        task_type="generation",
        examples=[
            TaskExample(
                input_text="What do you do with a pencil? ",
                expected_output="Write or draw with it."
            ),
            TaskExample(
                input_text="How do you cool down a hot drink? ",
                expected_output="Blow on it or wait for it to cool."
            ),
            TaskExample(
                input_text="Why don't people walk on their hands? ",
                expected_output="It's difficult to balance and inefficient."
            ),
            TaskExample(
                input_text="What happens when ice gets warm? ",
                expected_output="It melts into water."
            ),
            TaskExample(
                input_text="Why do we wear sunglasses? ",
                expected_output="To protect our eyes from the sun."
            ),
        ]
    )
    
    # Summarization task
    summarization = TaskConfig(
        name="summarization",
        description="Text summarization",
        metric="accuracy",  # Should be ROUGE in production
        task_type="generation",
        examples=[
            TaskExample(
                input_text="The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow models to weigh the importance of different words in a sequence. This innovation enabled more parallelization during training compared to previous recurrent neural network approaches. Transformers have become the foundation for models like BERT, GPT, and T5, which have achieved state-of-the-art results across a wide range of language tasks. The scalability of transformers has also led to increasingly large models with billions of parameters. Summarize: ",
                expected_output="Transformer architecture revolutionized NLP with self-attention mechanisms, enabling better parallelization than RNNs and becoming the foundation for models like BERT and GPT."
            ),
            TaskExample(
                input_text="Neural plasticity refers to the brain's ability to modify its connections and reorganize itself throughout life. This process allows for learning new information, acquiring skills, and recovering from brain injuries. Plasticity occurs through various mechanisms, including the strengthening or weakening of synapses, the formation of new neural connections, and the pruning of unused pathways. Environmental enrichment, training, and experience all contribute to neural plasticity. Research has shown that plasticity is greater during critical periods in development but continues throughout adulthood. Summarize: ",
                expected_output="Neural plasticity is the brain's ability to reorganize itself by forming new connections throughout life, enabling learning, skill acquisition, and recovery from injuries through mechanisms like synaptic strengthening and pruning."
            ),
            TaskExample(
                input_text="Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning. Supervised learning involves training on labeled data to make predictions or classifications. Unsupervised learning identifies patterns in unlabeled data. Reinforcement learning trains agents through a system of rewards and penalties. Each approach has different applications: supervised learning is used for tasks like spam detection and image recognition, unsupervised learning for clustering and anomaly detection, and reinforcement learning for robotics and game playing. The choice of algorithm depends on the available data and the specific problem being addressed. Summarize: ",
                expected_output="Machine learning algorithms fall into three categories: supervised learning uses labeled data for predictions, unsupervised learning finds patterns in unlabeled data, and reinforcement learning trains agents through rewards and penalties."
            ),
        ]
    )
    
    # Code completion task
    code_completion = TaskConfig(
        name="code_completion",
        description="Python code completion",
        metric="accuracy",
        task_type="generation",
        examples=[
            TaskExample(
                input_text="def calculate_average(numbers):\n    \"\"\"Calculate the average of a list of numbers.\"\"\"\n    ",
                expected_output="if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)"
            ),
            TaskExample(
                input_text="def is_palindrome(text):\n    \"\"\"Check if a string is a palindrome.\"\"\"\n    ",
                expected_output="text = text.lower()\n    text = ''.join(c for c in text if c.isalnum())\n    return text == text[::-1]"
            ),
            TaskExample(
                input_text="def find_max(numbers):\n    \"\"\"Find the maximum value in a list.\"\"\"\n    ",
                expected_output="if not numbers:\n        return None\n    max_val = numbers[0]\n    for num in numbers[1:]:\n        if num > max_val:\n            max_val = num\n    return max_val"
            ),
        ]
    )
    
    # Natural language inference task
    nli_task = TaskConfig(
        name="nli",
        description="Natural language inference",
        metric="accuracy",
        task_type="classification",
        examples=[
            TaskExample(
                input_text="Premise: The man is playing a guitar.\nHypothesis: The man is making music.",
                expected_output="0"  # Entailment
            ),
            TaskExample(
                input_text="Premise: The child is playing outside.\nHypothesis: The child is sleeping.",
                expected_output="2"  # Contradiction
            ),
            TaskExample(
                input_text="Premise: The woman is walking down the street.\nHypothesis: The woman is going to the store.",
                expected_output="1"  # Neutral
            ),
        ]
    )
    
    # Create task suite
    return TaskSuite(
        name="diverse_tasks",
        tasks=[commonsense_qa, summarization, code_completion, nli_task]
    )


def create_memory_stress_task() -> TaskSuite:
    """
    Create a task designed to stress model memory with long contexts.
    
    Returns:
        TaskSuite with memory-intensive tasks
    """
    # Long context completion
    long_context = TaskConfig(
        name="long_context",
        description="Long context completion with relevant information at the beginning",
        metric="accuracy",
        task_type="generation",
        max_input_length=1024,
        examples=[
            TaskExample(
                input_text="The capital of France is Paris. " + "Irrelevant filler text. " * 50 + "What is the capital of France? ",
                expected_output="Paris"
            ),
            TaskExample(
                input_text="The primary colors are red, blue, and yellow. " + "Irrelevant filler text. " * 50 + "List the primary colors: ",
                expected_output="Red, blue, and yellow."
            ),
            TaskExample(
                input_text="The speed of light is approximately 299,792,458 meters per second. " + "Irrelevant filler text. " * 50 + "What is the speed of light? ",
                expected_output="299,792,458 meters per second"
            ),
        ]
    )
    
    # Key-value recall
    key_value_recall = TaskConfig(
        name="key_value_recall",
        description="Remember multiple key-value pairs scattered throughout the context",
        metric="accuracy",
        task_type="generation",
        max_input_length=1024,
        examples=[
            TaskExample(
                input_text="Alice: 42\nFiller text.\nBob: 17\nFiller text.\nCharlie: 33\nFiller text.\nDavid: 29\nFiller text.\nWhat is Charlie's number? ",
                expected_output="33"
            ),
            TaskExample(
                input_text="Red: Apple\nFiller text.\nBlue: Sky\nFiller text.\nGreen: Grass\nFiller text.\nYellow: Sun\nFiller text.\nWhat is associated with Blue? ",
                expected_output="Sky"
            ),
            TaskExample(
                input_text="January: Winter\nFiller text.\nApril: Spring\nFiller text.\nJuly: Summer\nFiller text.\nOctober: Fall\nFiller text.\nWhat season is July? ",
                expected_output="Summer"
            ),
        ]
    )
    
    # Create task suite
    return TaskSuite(
        name="memory_stress",
        tasks=[long_context, key_value_recall]
    )


def create_conflicting_tasks() -> TaskSuite:
    """
    Create tasks with conflicting objectives to stress neural plasticity.
    
    Returns:
        TaskSuite with conflicting tasks
    """
    # Standard text completion
    standard_completion = TaskConfig(
        name="standard_completion",
        description="Standard text completion according to training distribution",
        metric="perplexity",
        task_type="lm",
        examples=[
            TaskExample(
                input_text="The capital of France is",
                expected_output=" Paris."
            ),
            TaskExample(
                input_text="Machine learning is a subset of",
                expected_output=" artificial intelligence."
            ),
            TaskExample(
                input_text="The earth revolves around the",
                expected_output=" sun."
            ),
        ]
    )
    
    # Reversed text completion
    reversed_completion = TaskConfig(
        name="reversed_completion",
        description="Complete with the reverse of normal expectation",
        metric="accuracy",
        task_type="generation",
        examples=[
            TaskExample(
                input_text="The opposite of hot is",
                expected_output="cold"
            ),
            TaskExample(
                input_text="The opposite of day is",
                expected_output="night"
            ),
            TaskExample(
                input_text="The opposite of good is",
                expected_output="bad"
            ),
        ]
    )
    
    # Literal task
    literal_task = TaskConfig(
        name="literal_task",
        description="Complete with literal interpretation",
        metric="accuracy",
        task_type="generation",
        examples=[
            TaskExample(
                input_text="Describe what a 'raining cats and dogs' weather looks like literally: ",
                expected_output="Actual cats and dogs falling from the sky."
            ),
            TaskExample(
                input_text="What would 'I'm all ears' look like literally? ",
                expected_output="A person whose body consists entirely of ears."
            ),
            TaskExample(
                input_text="Describe what 'break a leg' means literally: ",
                expected_output="Actually fracturing one's leg bone."
            ),
        ]
    )
    
    # Idiomatic task
    idiomatic_task = TaskConfig(
        name="idiomatic_task",
        description="Complete with idiomatic interpretation",
        metric="accuracy",
        task_type="generation",
        examples=[
            TaskExample(
                input_text="What does 'raining cats and dogs' actually mean? ",
                expected_output="It's raining very heavily."
            ),
            TaskExample(
                input_text="What does 'I'm all ears' actually mean? ",
                expected_output="I'm listening attentively."
            ),
            TaskExample(
                input_text="What does 'break a leg' actually mean? ",
                expected_output="Good luck (especially in a performance)."
            ),
        ]
    )
    
    # Create task suite
    return TaskSuite(
        name="conflicting_tasks",
        tasks=[standard_completion, reversed_completion, literal_task, idiomatic_task]
    )


if __name__ == "__main__":
    # Example usage
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Create task suite
    task_suite = create_diverse_task_suite()
    
    # Load model and tokenizer
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataloader
    dataloader = task_suite.create_dataloader("commonsense_qa", tokenizer)
    
    # Evaluate on task
    metrics = task_suite.evaluate("commonsense_qa", model, tokenizer)
    print(f"Metrics: {metrics}")