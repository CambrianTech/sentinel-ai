# Continuous Training Setup

## What's Running

Two GPT-2 models training continuously in separate tmux sessions:

1. **UNPRUNED** - Baseline GPT-2 (124M params, 144 heads)
2. **PRUNED** - GPT-2 with 40% magnitude pruning (86M params, 86 heads)

Both models:
- Train on WikiText-2 dataset
- Generate text samples every epoch
- Save checkpoints automatically
- Can resume from checkpoints if interrupted

## Quick Commands

```bash
# Start both models (or resume from checkpoints)
./start_training.sh

# Monitor progress (auto-updating dashboard)
./monitor_training.sh

# View text samples (live)
./start_training.sh samples unpruned
./start_training.sh samples pruned

# View training logs (live)
./start_training.sh logs unpruned
./start_training.sh logs pruned

# Attach to training session (see live output)
tmux attach -t sentinel-unpruned   # Ctrl+B then D to detach
tmux attach -t sentinel-pruned

# List active sessions
./start_training.sh list

# Stop all training
./start_training.sh kill
```

## Directory Structure

```
checkpoints/
├── gpt2_unpruned_wikitext/
│   ├── checkpoints/           # Model checkpoints
│   │   ├── checkpoint_epoch_1.pt
│   │   ├── checkpoint_epoch_2.pt
│   │   └── best_model.pt
│   ├── text_samples.txt       # Generated text samples
│   ├── training_metrics.jsonl # Training metrics (loss, LR)
│   ├── config.json            # Training configuration
│   └── training.log           # Full training log
│
└── gpt2_pruned_wikitext/
    └── (same structure)
```

## Monitoring

### Real-Time Dashboard
```bash
./monitor_training.sh
```
Shows:
- Active sessions
- Latest training steps and loss
- Recent text samples
- Updates every 10 seconds

### Sample Output
Watch text quality improve over time:
```bash
# Epoch 1 - Random
→ The future of artificial intelligence sponge 374 roofDRIVE Gate...