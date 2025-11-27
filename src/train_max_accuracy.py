"""
Maximum accuracy training script with advanced techniques.

Optimized for RTX 3090 and best possible CommonsenseQA performance.
Uses gradient accumulation, learning rate warmup, and careful hyperparameters.
"""

import sys
from config import TrainingConfig
from train import main as train_main

# Maximum accuracy configuration for RTX 3090
config = TrainingConfig(
    # Model
    model_name='bert-base-uncased',
    max_length=256,
    num_choices=5,
    dropout_prob=0.1,

    # Stage 1: SWAG fine-tuning (more stable than HellaSwag)
    stage1_dataset='swag',
    stage1_epochs=4,  # More epochs for thorough learning
    stage1_batch_size=4,  # Small batch for better generalization
    stage1_learning_rate=1.5e-5,  # Conservative LR
    stage1_warmup_steps=1000,  # More warmup for stability
    stage1_output_dir='checkpoints/max_accuracy_stage1',

    # Stage 2: CommonsenseQA fine-tuning
    stage2_dataset='commonsenseqa',
    stage2_epochs=8,  # Plenty of epochs
    stage2_batch_size=4,  # Small batch for best generalization
    stage2_learning_rate=8e-6,  # Lower LR for careful tuning
    stage2_warmup_steps=500,  # Warmup for smooth start
    stage2_output_dir='checkpoints/max_accuracy_final',

    # Training optimizations
    gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
    max_grad_norm=1.0,  # Gradient clipping for stability
    weight_decay=0.01,  # L2 regularization
    adam_epsilon=1e-8,

    # Evaluation and checkpointing
    eval_steps=500,
    save_steps=500,
    logging_steps=50,

    # Hardware
    device='cuda',
    fp16=False,  # Set to True if you want 2x memory savings (may reduce accuracy slightly)

    # Misc
    seed=42,
    skip_stage1=False,
    stage1_checkpoint=None
)

if __name__ == '__main__':
    print("="*70)
    print("MAXIMUM ACCURACY TRAINING MODE")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Hardware: RTX 3090 (24GB VRAM)")
    print(f"  Strategy: Small batch ({config.stage2_batch_size}) + Gradient accumulation ({config.gradient_accumulation_steps})")
    print(f"  Effective batch size: {config.stage2_batch_size * config.gradient_accumulation_steps}")
    print(f"  Stage 1: {config.stage1_epochs} epochs on {config.stage1_dataset.upper()}")
    print(f"  Stage 2: {config.stage2_epochs} epochs on CommonsenseQA")
    print(f"\nExpected training time: ~10-12 hours")
    print(f"Target accuracy: 67-69%")
    print("="*70)
    print()

    # Run training with optimized config
    train_main(config)
