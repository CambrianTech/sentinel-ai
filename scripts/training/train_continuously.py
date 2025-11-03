#!/usr/bin/env python3
"""
Continuous Training for Sentinel-AI - Compare Unpruned vs Pruned Models

Features:
- Train unpruned OR pruned model
- Resume from checkpoints automatically
- Multiple dataset options (WikiText, OpenWebText, TinyStories)
- Generates text samples every epoch
- Saves checkpoints automatically
- Runs in tmux for long-running training

Usage:
    # Train unpruned model
    python train_continuously.py --model gpt2 --mode unpruned

    # Train pruned model (40% magnitude pruning)
    python train_continuously.py --model gpt2 --mode pruned --pruning-level 0.4

    # Use different dataset
    python train_continuously.py --dataset openwebtext --mode pruned

    # Resume automatically from checkpoints
    python train_continuously.py --model gpt2 --mode pruned  # Will find and resume
"""

import sys
import os
sys.path.insert(0, '/Volumes/FlashGordon/cambrian/sentinel-ai')
os.chdir('/Volumes/FlashGordon/cambrian/sentinel-ai')

import argparse
import torch
import json
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

from models.loaders.loader import load_baseline_model, load_adaptive_model
from sentinel.pruning.entropy_magnitude import magnitude_based_pruning


DATASETS = {
    'wikitext': {
        'name': 'wikitext',
        'config': 'wikitext-2-raw-v1',
        'split': 'train',
        'description': 'WikiText-2: High quality Wikipedia articles'
    },
    'openwebtext': {
        'name': 'openwebtext',
        'config': None,
        'split': 'train',
        'description': 'OpenWebText: Reddit links with high karma'
    },
    'tinystories': {
        'name': 'roneneldan/TinyStories',
        'config': None,
        'split': 'train',
        'description': 'TinyStories: Short stories for language models'
    }
}


class ContinuousTrainer:
    def __init__(self, args):
        self.args = args

        # Setup paths
        mode_suffix = "pruned" if args.mode == "pruned" else "unpruned"
        self.output_dir = Path(f"./checkpoints/{args.model}_{mode_suffix}_{args.dataset}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.samples_file = self.output_dir / "text_samples.txt"
        self.metrics_file = self.output_dir / "training_metrics.jsonl"
        self.config_file = self.output_dir / "config.json"

        # Training config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        # Save config
        self.save_config()

        self.print_header()

    def save_config(self):
        """Save training configuration."""
        config = {
            'model': self.args.model,
            'mode': self.args.mode,
            'pruning_level': self.args.pruning_level,
            'dataset': self.args.dataset,
            'batch_size': self.args.batch_size,
            'gradient_accumulation': self.args.gradient_accumulation,
            'max_length': self.args.max_length,
            'learning_rate': self.args.learning_rate,
            'created': datetime.now().isoformat()
        }

        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def print_header(self):
        """Print training configuration."""
        print("="*80)
        print(f"CONTINUOUS TRAINING - SENTINEL-AI")
        print("="*80)

        dataset_info = DATASETS[self.args.dataset]
        pruning_info = f"{self.args.pruning_level*100}% MAGNITUDE" if self.args.mode == "pruned" else "NONE"

        print(f"\n📋 Configuration:")
        print(f"   Model: {self.args.model}")
        print(f"   Mode: {self.args.mode.upper()}")
        print(f"   Pruning: {pruning_info}")
        print(f"   Dataset: {self.args.dataset} ({dataset_info['description']})")
        print(f"   Device: {self.device}")
        print(f"   Output: {self.output_dir}")
        print(f"   Batch size: {self.args.batch_size}")
        print(f"   Gradient accumulation: {self.args.gradient_accumulation}")
        print(f"   Effective batch size: {self.args.batch_size * self.args.gradient_accumulation}")
        print(f"   Max length: {self.args.max_length}")
        print(f"   Learning rate: {self.args.learning_rate}")

    def load_or_create_model(self):
        """Load from checkpoint or create new model."""
        print("\n" + "="*80)
        print("MODEL SETUP")
        print("="*80)

        latest_checkpoint = self.find_latest_checkpoint()

        if latest_checkpoint:
            print(f"\n✅ Found checkpoint: {latest_checkpoint.name}")
            return self.load_checkpoint(latest_checkpoint)
        else:
            print(f"\n📦 No checkpoint found - creating new {self.args.mode} model...")
            return self.create_model()

    def find_latest_checkpoint(self):
        """Find the most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def load_checkpoint(self, checkpoint_path):
        """Load model and training state from checkpoint."""
        print(f"\n📂 Loading checkpoint...")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Global step: {checkpoint['global_step']}")
        print(f"   Best loss: {checkpoint['best_loss']:.4f}")
        print(f"   Loss: {checkpoint['loss']:.4f}")

        # Recreate model architecture
        baseline_model = load_baseline_model(self.args.model, self.device)

        if self.args.mode == "pruned":
            # For pruned models, we need to prune again then load weights
            model = load_adaptive_model(self.args.model, baseline_model, self.device, quiet=True)
            # Note: In production, we'd save pruning masks and restore them
            # For now, we'll load the full state dict
        else:
            model = baseline_model

        model.load_state_dict(checkpoint['model_state_dict'])

        # Restore training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        print("\n✅ Checkpoint loaded - resuming training")

        return model, checkpoint.get('optimizer_state_dict'), checkpoint.get('scheduler_state_dict')

    def create_model(self):
        """Create a new model (unpruned or pruned)."""
        print("\n1. Loading baseline model...")
        baseline_model = load_baseline_model(self.args.model, self.device)

        if self.args.mode == "unpruned":
            print("\n✅ Using UNPRUNED model (baseline)")
            return baseline_model, None, None

        # Pruned mode
        print("\n2. Creating adaptive model...")
        adaptive_model = load_adaptive_model(self.args.model, baseline_model, self.device, quiet=True)

        print("\n3. Preparing calibration data...")
        tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        tokenizer.pad_token = tokenizer.eos_token

        # Load dataset for calibration
        dataset_info = DATASETS[self.args.dataset]
        if dataset_info['config']:
            dataset = load_dataset(dataset_info['name'], dataset_info['config'], split="train[:1000]")
        else:
            dataset = load_dataset(dataset_info['name'], split="train[:1000]")

        texts = [text for text in dataset['text'] if len(text.strip()) > 50][:100]

        calibration_data = []
        for text in texts[:50]:
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=128, padding="max_length")
            calibration_data.append(tokens['input_ids'])

        calibration_tensor = torch.cat(calibration_data, dim=0).to(self.device)
        dataset_calib = TensorDataset(calibration_tensor)
        dataloader = DataLoader(dataset_calib, batch_size=4, shuffle=False)

        print(f"\n4. Pruning with MAGNITUDE strategy ({self.args.pruning_level*100}%)...")
        print("   (Magnitude pruning showed 26% improvement!)")

        pruned_heads = magnitude_based_pruning(
            adaptive_model,
            prune_ratio=self.args.pruning_level
        )

        pruned_model = adaptive_model  # Model is pruned in-place

        print("\n✅ Pruned model created")

        return pruned_model, None, None

    def prepare_training_data(self):
        """Load and prepare training dataset."""
        print("\n" + "="*80)
        print("TRAINING DATA")
        print("="*80)

        tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        tokenizer.pad_token = tokenizer.eos_token

        # Load dataset
        dataset_info = DATASETS[self.args.dataset]
        print(f"\n📚 Loading {dataset_info['description']}...")

        if dataset_info['config']:
            dataset = load_dataset(dataset_info['name'], dataset_info['config'], split=dataset_info['split'])
        else:
            dataset = load_dataset(dataset_info['name'], split=dataset_info['split'])

        # Filter texts
        texts = [text for text in dataset['text'] if len(text.strip()) > 100]

        # Limit dataset size based on user preference
        max_samples = min(self.args.max_samples, len(texts))
        texts = texts[:max_samples]

        print(f"   Total texts: {len(texts)}")

        # Tokenize
        print(f"\n⚙️  Tokenizing (max_length={self.args.max_length})...")
        all_input_ids = []

        for i, text in enumerate(texts):
            if i % 1000 == 0 and i > 0:
                print(f"   Processed {i}/{len(texts)} texts...")

            tokens = tokenizer(
                text,
                truncation=True,
                max_length=self.args.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            all_input_ids.append(tokens['input_ids'])

        # Create dataset
        input_tensor = torch.cat(all_input_ids, dim=0)
        dataset = TensorDataset(input_tensor, input_tensor)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True
        )

        print(f"\n✅ Training data ready:")
        print(f"   Samples: {len(dataset):,}")
        print(f"   Batches per epoch: {len(dataloader):,}")
        print(f"   Tokens per sample: {self.args.max_length}")

        return dataloader, tokenizer

    def generate_samples(self, model, tokenizer, epoch):
        """Generate text samples."""
        model.eval()

        prompts = [
            "The future of artificial intelligence",
            "In a world where technology",
            "Scientists have discovered",
            "The most important lesson",
            "Deep learning models"
        ]

        samples = []
        samples.append(f"\n{'='*80}")
        samples.append(f"TEXT SAMPLES - Epoch {epoch} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        samples.append(f"Mode: {self.args.mode.upper()} | Best Loss: {self.best_loss:.4f}")
        samples.append(f"{'='*80}\n")

        for i, prompt in enumerate(prompts, 1):
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

            sample_text = f"{i}. \"{prompt}\"\n   → {generated}\n"
            samples.append(sample_text)
            print(sample_text)

        samples.append("="*80 + "\n")

        with open(self.samples_file, 'a') as f:
            f.write('\n'.join(samples))

        model.train()

    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_loss': self.best_loss,
            'loss': loss,
            'config': vars(self.args),
            'timestamp': datetime.now().isoformat()
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"   💾 Checkpoint: {checkpoint_path.name}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"   ⭐ NEW BEST MODEL!")

        # Keep last 5 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        for old_checkpoint in checkpoints[:-5]:
            old_checkpoint.unlink()

    def log_metrics(self, epoch, step, loss, lr):
        """Log training metrics."""
        metrics = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'lr': lr,
            'mode': self.args.mode,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def train(self):
        """Main training loop."""
        # Setup
        model_result = self.load_or_create_model()
        model, optimizer_state, scheduler_state = model_result if len(model_result) == 3 else (model_result, None, None)

        dataloader, tokenizer = self.prepare_training_data()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        total_steps = len(dataloader) * 1000
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps
        )
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)

        print("\n" + "="*80)
        print("🚀 STARTING TRAINING")
        print("="*80)
        print(f"\n💡 Tips:")
        print(f"   - Training runs until interrupted (Ctrl+C)")
        print(f"   - Checkpoints: {self.checkpoint_dir}")
        print(f"   - Samples: {self.samples_file}")
        print(f"   - Metrics: {self.metrics_file}")
        print(f"   - Resume: Just run this script again!")
        print(f"\n📊 Monitor with:")
        print(f"   tail -f {self.samples_file}")
        print(f"   tail -f {self.output_dir}/training.log")
        print("\n")

        model.train()

        try:
            while True:
                self.epoch += 1
                epoch_loss = 0.0
                num_batches = 0

                print(f"\n{'='*80}")
                print(f"📈 EPOCH {self.epoch}")
                print(f"{'='*80}\n")

                for batch_idx, (input_ids, labels) in enumerate(dataloader):
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss / self.args.gradient_accumulation

                    loss.backward()

                    if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        self.global_step += 1

                        if self.global_step % 10 == 0:
                            current_loss = loss.item() * self.args.gradient_accumulation
                            current_lr = scheduler.get_last_lr()[0]
                            print(f"   Step {self.global_step:,} | Loss: {current_loss:.4f} | LR: {current_lr:.2e}")
                            self.log_metrics(self.epoch, self.global_step, current_loss, current_lr)

                    epoch_loss += loss.item() * self.args.gradient_accumulation
                    num_batches += 1

                avg_loss = epoch_loss / num_batches
                print(f"\n✅ Epoch {self.epoch} | Avg Loss: {avg_loss:.4f}")

                print(f"\n📝 Generating samples...\n")
                self.generate_samples(model, tokenizer, self.epoch)

                is_best = avg_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_loss

                print(f"\n💾 Saving...")
                self.save_checkpoint(model, optimizer, scheduler, self.epoch, avg_loss, is_best)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted - saving...")
            self.save_checkpoint(model, optimizer, scheduler, self.epoch, epoch_loss / max(num_batches, 1))
            print("   ✅ Saved!")


def main():
    parser = argparse.ArgumentParser(description='Continuous training for Sentinel-AI')

    # Model options
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'distilgpt2', 'gpt2-medium'],
                       help='Model to train')
    parser.add_argument('--mode', default='pruned', choices=['unpruned', 'pruned'],
                       help='Train unpruned or pruned model')
    parser.add_argument('--pruning-level', type=float, default=0.4,
                       help='Pruning level (0.0-1.0)')

    # Dataset options
    parser.add_argument('--dataset', default='wikitext', choices=list(DATASETS.keys()),
                       help='Training dataset')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Maximum training samples')

    # Training options
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=100,
                       help='Warmup steps')

    args = parser.parse_args()

    trainer = ContinuousTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
