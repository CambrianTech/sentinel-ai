# Pruning API

This API provides simple functions for pruning transformer models and fine-tuning them to recover performance.

## Core API Functions

### pruning.py
- `compute_head_importance(model, dataloader, num_batches=10, device="cuda")` - Compute importance scores for each attention head using a specified strategy
- `prune_heads(model, importance, pruning_percent=0.3, device="cuda")` - Prune the least important attention heads by creating a head mask
- `fine_tune(model, train_dataloader, val_dataloader, num_epochs=3, learning_rate=5e-5, max_steps=None, device="cuda", eval_every=100, callbacks=None)` - Fine-tune a pruned model to recover performance
- `evaluate_model(model, dataloader, device="cuda")` - Evaluate a model on the given dataloader

### data.py
- `load_wikitext()` - Load Wikitext dataset for training and evaluation
- `prepare_data(tokenizer, text_data, max_length=512, batch_size=4)` - Prepare dataset for training/evaluation
- `prepare_test_data(tokenizer, max_length=512, batch_size=4, num_samples=10)` - Create a tiny test dataset for quick testing

## Usage

```python
from utils.pruning.api.pruning import compute_head_importance, prune_heads, fine_tune, evaluate_model
from utils.pruning.api.data import load_wikitext, prepare_data, prepare_test_data

# Load model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Load data
train_data, val_data = load_wikitext()
train_dataloader = prepare_data(tokenizer, train_data, batch_size=4)
val_dataloader = prepare_data(tokenizer, val_data, batch_size=4)

# Evaluate initial model
initial_loss, initial_ppl = evaluate_model(model, val_dataloader)
print(f"Initial model - Loss: {initial_loss:.4f}, Perplexity: {initial_ppl:.2f}")

# Compute head importance
importance = compute_head_importance(model, val_dataloader)

# Prune heads
pruned_heads = prune_heads(model, importance, pruning_percent=0.3)

# Evaluate pruned model
pruned_loss, pruned_ppl = evaluate_model(model, val_dataloader)
print(f"Pruned model - Loss: {pruned_loss:.4f}, Perplexity: {pruned_ppl:.2f}")

# Fine-tune the pruned model
final_loss, final_ppl = fine_tune(model, train_dataloader, val_dataloader, num_epochs=3)
print(f"Fine-tuned model - Loss: {final_loss:.4f}, Perplexity: {final_ppl:.2f}")
```

## Testing

To verify the API works correctly, run:

```bash
python scripts/test_pruning_api.py --test --device cpu
```

This will run a quick test using a tiny dataset and verify all API functions are working correctly.

For more extensive testing with real data, omit the `--test` flag:

```bash
python scripts/test_pruning_api.py --device cuda
```