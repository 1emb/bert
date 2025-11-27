"""
Two-stage training script for BERT on CommonsenseQA.

Stage 1: Fine-tune on intermediate dataset (SWAG or HellaSwag)
Stage 2: Fine-tune on CommonsenseQA
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse
import json

from model import BertForMultipleChoice
from data_loader import get_dataloader
from config import TrainingConfig


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, scheduler, device, max_grad_norm, gradient_accumulation_steps):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits, loss = model(input_ids, attention_mask, labels)

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (step + 1),
            'acc': correct / total
        })

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits, loss = model(input_ids, attention_mask, labels)

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def train_stage(
    model,
    train_dataloader,
    val_dataloader,
    config,
    stage_name,
    output_dir,
    epochs,
    learning_rate,
    warmup_steps
):
    """Train a single stage."""
    print(f"\n{'='*50}")
    print(f"Starting {stage_name}")
    print(f"{'='*50}\n")

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay
    )

    total_steps = len(train_dataloader) * epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    best_val_acc = 0
    training_history = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_epoch(
            model, train_dataloader, optimizer, scheduler, device,
            config.max_grad_norm, config.gradient_accumulation_steps
        )

        val_loss, val_acc = evaluate(model, val_dataloader, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")

    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n{stage_name} completed. Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc


def main(config: TrainingConfig):
    """Main training function."""
    set_seed(config.seed)

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Stage 1: Intermediate fine-tuning
    if not config.skip_stage1:
        print("\n" + "="*70)
        print("STAGE 1: Intermediate Fine-tuning")
        print("="*70)

        # Load stage 1 data
        print(f"Loading {config.stage1_dataset} dataset...")
        stage1_train_loader = get_dataloader(
            config.stage1_dataset,
            tokenizer,
            'train',
            batch_size=config.stage1_batch_size,
            max_length=config.max_length,
            shuffle=True
        )

        stage1_val_loader = get_dataloader(
            config.stage1_dataset,
            tokenizer,
            'validation',
            batch_size=config.stage1_batch_size,
            max_length=config.max_length,
            shuffle=False
        )

        # Initialize model
        print("Initializing model...")
        num_choices = 4 if config.stage1_dataset.lower() == 'swag' else config.num_choices
        model = BertForMultipleChoice(
            model_name=config.model_name,
            num_choices=num_choices,
            dropout_prob=config.dropout_prob
        )

        # Train stage 1
        train_stage(
            model,
            stage1_train_loader,
            stage1_val_loader,
            config,
            "Stage 1: " + config.stage1_dataset.upper(),
            config.stage1_output_dir,
            config.stage1_epochs,
            config.stage1_learning_rate,
            config.stage1_warmup_steps
        )

        stage1_checkpoint = config.stage1_output_dir
    else:
        print("\nSkipping Stage 1 (using provided checkpoint)")
        stage1_checkpoint = config.stage1_checkpoint

    # Stage 2: CommonsenseQA fine-tuning
    print("\n" + "="*70)
    print("STAGE 2: CommonsenseQA Fine-tuning")
    print("="*70)

    # Load CommonsenseQA data
    print("Loading CommonsenseQA dataset...")
    stage2_train_loader = get_dataloader(
        'commonsenseqa',
        tokenizer,
        'train',
        batch_size=config.stage2_batch_size,
        max_length=config.max_length,
        shuffle=True
    )

    stage2_val_loader = get_dataloader(
        'commonsenseqa',
        tokenizer,
        'validation',
        batch_size=config.stage2_batch_size,
        max_length=config.max_length,
        shuffle=False
    )

    # Load model from stage 1
    if stage1_checkpoint and os.path.exists(stage1_checkpoint):
        print(f"Loading model from {stage1_checkpoint}...")
        model = BertForMultipleChoice.from_pretrained(
            stage1_checkpoint,
            num_choices=config.num_choices
        )
    else:
        print("Starting from base BERT model...")
        model = BertForMultipleChoice(
            model_name=config.model_name,
            num_choices=config.num_choices,
            dropout_prob=config.dropout_prob
        )

    # Train stage 2
    best_acc = train_stage(
        model,
        stage2_train_loader,
        stage2_val_loader,
        config,
        "Stage 2: CommonsenseQA",
        config.stage2_output_dir,
        config.stage2_epochs,
        config.stage2_learning_rate,
        config.stage2_warmup_steps
    )

    print("\n" + "="*70)
    print("TWO-STAGE TRAINING COMPLETED")
    print("="*70)
    print(f"Final CommonsenseQA validation accuracy: {best_acc:.4f}")
    print(f"Final model saved to: {config.stage2_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-stage BERT fine-tuning for CommonsenseQA')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=256)

    # Stage 1 arguments
    parser.add_argument('--stage1_dataset', type=str, default='swag',
                        choices=['swag', 'hellaswag'])
    parser.add_argument('--stage1_epochs', type=int, default=3)
    parser.add_argument('--stage1_batch_size', type=int, default=8)
    parser.add_argument('--stage1_lr', type=float, default=2e-5)
    parser.add_argument('--stage1_output_dir', type=str, default='checkpoints/stage1')
    parser.add_argument('--skip_stage1', action='store_true',
                        help='Skip stage 1 training')
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                        help='Path to stage 1 checkpoint (if skipping stage 1)')

    # Stage 2 arguments
    parser.add_argument('--stage2_epochs', type=int, default=5)
    parser.add_argument('--stage2_batch_size', type=int, default=8)
    parser.add_argument('--stage2_lr', type=float, default=1e-5)
    parser.add_argument('--stage2_output_dir', type=str, default='checkpoints/stage2')

    # Training arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Create config from arguments
    config = TrainingConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        stage1_dataset=args.stage1_dataset,
        stage1_epochs=args.stage1_epochs,
        stage1_batch_size=args.stage1_batch_size,
        stage1_learning_rate=args.stage1_lr,
        stage1_output_dir=args.stage1_output_dir,
        stage2_epochs=args.stage2_epochs,
        stage2_batch_size=args.stage2_batch_size,
        stage2_learning_rate=args.stage2_lr,
        stage2_output_dir=args.stage2_output_dir,
        seed=args.seed,
        device=args.device,
        skip_stage1=args.skip_stage1,
        stage1_checkpoint=args.stage1_checkpoint
    )

    main(config)
