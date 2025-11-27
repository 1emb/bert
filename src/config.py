"""
Configuration for training.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for two-stage training."""

    # Model
    model_name: str = 'bert-base-uncased'
    max_length: int = 256
    num_choices: int = 5
    dropout_prob: float = 0.1

    # Stage 1: Intermediate fine-tuning
    stage1_dataset: str = 'swag'  # Options: 'swag', 'hellaswag'
    stage1_epochs: int = 3
    stage1_batch_size: int = 8
    stage1_learning_rate: float = 2e-5
    stage1_warmup_steps: int = 500
    stage1_output_dir: str = 'checkpoints/stage1'

    # Stage 2: CommonsenseQA fine-tuning
    stage2_dataset: str = 'commonsenseqa'
    stage2_epochs: int = 5
    stage2_batch_size: int = 8
    stage2_learning_rate: float = 1e-5
    stage2_warmup_steps: int = 200
    stage2_output_dir: str = 'checkpoints/stage2'

    # Training
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    seed: int = 42

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100

    # Hardware
    device: str = 'cuda'  # 'cuda' or 'cpu'
    fp16: bool = False

    # Misc
    skip_stage1: bool = False  # Set to True to skip stage 1
    stage1_checkpoint: Optional[str] = None  # Path to stage 1 checkpoint


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    model_path: str = 'checkpoints/stage2'
    dataset: str = 'commonsenseqa'
    split: str = 'validation'
    batch_size: int = 16
    max_length: int = 256
    device: str = 'cuda'
