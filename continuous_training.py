#!/usr/bin/env python3
"""
Continuous Training Script for Sentinel-AI

This script:
1. Trains a pruned model continuously with checkpointing
2. Generates text samples after each epoch
3. Saves checkpoints automatically
4. Can resume from last checkpoint if interrupted
5. Uses the BEST pruning strategy (magnitude - 26% improvement!)
"""

import sys
import os
sys.path.insert(0, '/Volumes/FlashGordon/cambrian/sentinel-ai')
os.chdir('/Volumes/FlashGordon/cambrian/sentinel-ai')

import torch
import json
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

from models.loaders.loader import load_baseline_model, load_adaptive_model
from sentinel.pruning.entropy_magnitude import magnitude_based_pruning


class ContinuousTrainer:
    def __init__(self, output_dir="./checkpoints/continuous_training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.samples_file = self.output_dir / "text_samples.txt"
        self.metrics_file = self.output_dir / "training_metrics.jsonl"

        self.model_name = "gpt2"  # 124M params - good size for 32GB RAM
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pruning_level = 0.4  # 40% - magnitude pruning works best!

        self.batch_size = 4
        self.gradient_accumulation_steps = 4
        self.max_length = 256
        self.learning_rate = 5e-5
        self.warmup_steps = 100

        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        print("="*80)
        print("CONTINUOUS TRAINING - SENTINEL-AI")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Pruning: {self.pruning_level*100}% (MAGNITUDE - best strategy)")
        print(f"  Output: {self.output_dir}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")

    def load_or_create_model(self):
        """Load from checkpoint or create new pruned model."""
        print("\n" + "="*80)
        print("LOADING/CREATING MODEL")
        print("="*80)

        latest_checkpoint = self.find_latest_checkpoint()

        if latest_checkpoint:
            print(f"\n✅ Found checkpoint: {latest_checkpoint}")
            print("   Resuming from checkpoint...")
            return self.load_checkpoint(latest_checkpoint)
        else:
            print("\n📦 No checkpoint found - creating new pruned model...")
            return self.create_pruned_model()

    def find_latest_checkpoint(self):
        """Find the most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def load_checkpoint(self, checkpoint_path):
        """Load model and training state from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        print(f"\n   Epoch: {checkpoint['epoch']}")
        print(f"   Global step: {checkpoint['global_step']}")
        print(f"   Best loss: {checkpoint['best_loss']:.4f}")

        # Load model
        baseline_model = load_baseline_model(self.model_name, self.device)
        model = load_adaptive_model(self.model_name, baseline_model, self.device, quiet=True)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Restore training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        print("\n✅ Checkpoint loaded successfully")

        return model, checkpoint.get('optimizer_state_dict'), checkpoint.get('scheduler_state_dict')

    def create_pruned_model(self):
        """Create a new pruned model using magnitude pruning."""
        print("\n1. Loading baseline model...")
        baseline_model = load_baseline_model(self.model_name, self.device)

        print("\n2. Creating adaptive model...")
        adaptive_model = load_adaptive_model(self.model_name, baseline_model, self.device, quiet=True)

        print("\n3. Preparing calibration data for pruning...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Use wikitext for calibration
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
        calibration_texts = [text for text in dataset['text'] if len(text.strip()) > 50][:100]

        calibration_data = []
        for text in calibration_texts[:50]:  # Use 50 samples for faster calibration
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=128, padding="max_length")
            calibration_data.append(tokens['input_ids'])

        calibration_tensor = torch.cat(calibration_data, dim=0).to(self.device)
        dataset_calib = TensorDataset(calibration_tensor)
        dataloader = DataLoader(dataset_calib, batch_size=4, shuffle=False)

        print(f"\n4. Pruning with MAGNITUDE strategy (40%)...")
        print("   (Magnitude pruning showed 26% improvement in experiments!)")

        pruned_model = magnitude_based_pruning(
            adaptive_model,
            dataloader,
            prune_ratio=self.pruning_level,
            device=self.device
        )

        print("\n✅ Pruned model created successfully")

        return pruned_model, None, None

    def prepare_training_data(self):
        """Load and prepare WikiText dataset."""
        print("\n" + "="*80)
        print("LOADING TRAINING DATA")
        print("="*80)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Load WikiText-2
        print("\nLoading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        # Filter out short texts
        texts = [text for text in dataset['text'] if len(text.strip()) > 100]

        print(f"   Total texts: {len(texts)}")

        # Tokenize
        print(f"\nTokenizing (max_length={self.max_length})...")
        all_input_ids = []

        for i, text in enumerate(texts[:5000]):  # Use first 5000 texts
            if i % 1000 == 0:
                print(f"   Processed {i}/{min(5000, len(texts))} texts...")

            tokens = tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            all_input_ids.append(tokens['input_ids'])

        # Create dataset
        input_tensor = torch.cat(all_input_ids, dim=0)
        dataset = TensorDataset(input_tensor, input_tensor)  # Labels = inputs for LM

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        print(f"\n✅ Training data prepared:")
        print(f"   Samples: {len(dataset)}")
        print(f"   Batches per epoch: {len(dataloader)}")
        print(f"   Tokens per sample: {self.max_length}")

        return dataloader, tokenizer

    def generate_samples(self, model, tokenizer, epoch):
        """Generate text samples to monitor quality."""
        model.eval()

        prompts = [
            "The future of artificial intelligence",
            "In a world where technology",
            "Scientists have discovered",
            "The most important thing to understand",
            "Recent advances in machine learning"
        ]

        samples = []
        samples.append(f"\n{'='*80}")
        samples.append(f"TEXT SAMPLES - Epoch {epoch} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

            sample_text = f"{i}. Prompt: \"{prompt}\"\n   Generated: {generated}\n"
            samples.append(sample_text)
            print(sample_text)

        samples.append("="*80 + "\n")

        # Save to file
        with open(self.samples_file, 'a') as f:
            f.write('\n'.join(samples))

        model.train()

    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_loss': self.best_loss,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"   💾 Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"   ⭐ New best model saved: {best_path}")

        # Keep only last 5 checkpoints (plus best)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()

    def log_metrics(self, epoch, step, loss, lr):
        """Log training metrics."""
        metrics = {
            'epoch': epoch,
            'global_step': step,
            'loss': loss,
            'learning_rate': lr,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def train(self):
        """Main training loop."""
        # Load or create model
        model_result = self.load_or_create_model()
        if len(model_result) == 3:
            model, optimizer_state, scheduler_state = model_result
        else:
            model = model_result
            optimizer_state = None
            scheduler_state = None

        # Prepare data
        dataloader, tokenizer = self.prepare_training_data()

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        # Setup scheduler
        total_steps = len(dataloader) * 1000  # Plan for 1000 epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)

        print("\n" + "="*80)
        print("STARTING CONTINUOUS TRAINING")
        print("="*80)
        print(f"\nTraining will run indefinitely until interrupted (Ctrl+C)")
        print(f"Checkpoints saved every epoch to: {self.checkpoint_dir}")
        print(f"Text samples logged to: {self.samples_file}")
        print(f"Metrics logged to: {self.metrics_file}")
        print("\nPress Ctrl+C to stop gracefully\n")

        model.train()

        try:
            while True:  # Train forever
                self.epoch += 1
                epoch_loss = 0.0
                num_batches = 0

                print(f"\n{'='*80}")
                print(f"EPOCH {self.epoch}")
                print(f"{'='*80}\n")

                for batch_idx, (input_ids, labels) in enumerate(dataloader):
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss / self.gradient_accumulation_steps

                    # Backward pass
                    loss.backward()

                    # Update weights
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        self.global_step += 1

                        # Log
                        if self.global_step % 10 == 0:
                            current_loss = loss.item() * self.gradient_accumulation_steps
                            current_lr = scheduler.get_last_lr()[0]
                            print(f"   Step {self.global_step} | Loss: {current_loss:.4f} | LR: {current_lr:.2e}")
                            self.log_metrics(self.epoch, self.global_step, current_loss, current_lr)

                    epoch_loss += loss.item() * self.gradient_accumulation_steps
                    num_batches += 1

                # End of epoch
                avg_loss = epoch_loss / num_batches
                print(f"\n📊 Epoch {self.epoch} complete - Avg Loss: {avg_loss:.4f}")

                # Generate samples
                print(f"\n📝 Generating text samples...\n")
                self.generate_samples(model, tokenizer, self.epoch)

                # Save checkpoint
                is_best = avg_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_loss

                print(f"\n💾 Saving checkpoint...")
                self.save_checkpoint(model, optimizer, scheduler, self.epoch, avg_loss, is_best)

                print(f"\n✅ Epoch {self.epoch} complete!")

        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            print("   Saving final checkpoint...")
            self.save_checkpoint(model, optimizer, scheduler, self.epoch, epoch_loss / max(num_batches, 1))
            print("   ✅ Final checkpoint saved")
            print(f"\n   To resume: Run this script again")
            print(f"   Checkpoints: {self.checkpoint_dir}")
            print(f"   Samples: {self.samples_file}")


def main():
    trainer = ContinuousTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
