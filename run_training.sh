#!/bin/bash

# Two-Stage BERT Fine-tuning for CommonsenseQA
# This script runs the complete two-stage training pipeline

echo "=========================================="
echo "Two-Stage BERT Fine-tuning"
echo "=========================================="
echo ""

# Configuration
STAGE1_DATASET="swag"  # Options: swag, hellaswag
STAGE1_EPOCHS=3
STAGE1_BATCH_SIZE=8
STAGE1_LR=2e-5

STAGE2_EPOCHS=5
STAGE2_BATCH_SIZE=8
STAGE2_LR=1e-5

DEVICE="cuda"  # Change to "cpu" if no GPU available
SEED=42

echo "Configuration:"
echo "  Stage 1 Dataset: $STAGE1_DATASET"
echo "  Stage 1 Epochs: $STAGE1_EPOCHS"
echo "  Stage 2 Epochs: $STAGE2_EPOCHS"
echo "  Device: $DEVICE"
echo ""

# Create output directories
mkdir -p checkpoints/stage1
mkdir -p checkpoints/stage2

# Run training
cd src

python train.py \
    --model_name bert-base-uncased \
    --stage1_dataset $STAGE1_DATASET \
    --stage1_epochs $STAGE1_EPOCHS \
    --stage1_batch_size $STAGE1_BATCH_SIZE \
    --stage1_lr $STAGE1_LR \
    --stage1_output_dir ../checkpoints/stage1 \
    --stage2_epochs $STAGE2_EPOCHS \
    --stage2_batch_size $STAGE2_BATCH_SIZE \
    --stage2_lr $STAGE2_LR \
    --stage2_output_dir ../checkpoints/stage2 \
    --device $DEVICE \
    --seed $SEED

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: checkpoints/stage2"
echo ""
echo "To evaluate the model, run:"
echo "  cd src && python evaluate.py --model_path ../checkpoints/stage2"
echo ""
