#!/bin/bash

# Two-Stage BERT Fine-tuning with Batch Size 16
# Optimized for GPUs with sufficient memory (16GB+)

echo "=========================================="
echo "Two-Stage BERT Fine-tuning (Batch Size 16)"
echo "=========================================="
echo ""

# Configuration with larger batch size
STAGE1_DATASET="swag"
STAGE1_EPOCHS=2  # Fewer epochs needed with larger batch
STAGE1_BATCH_SIZE=16
STAGE1_LR=3e-5  # Slightly higher LR for larger batch

STAGE2_EPOCHS=4  # Fewer epochs needed
STAGE2_BATCH_SIZE=16
STAGE2_LR=1.5e-5  # Scaled LR

DEVICE="cuda"
SEED=42

echo "Configuration:"
echo "  Stage 1 Dataset: $STAGE1_DATASET"
echo "  Stage 1 Epochs: $STAGE1_EPOCHS (fewer needed with larger batch)"
echo "  Stage 1 Batch Size: $STAGE1_BATCH_SIZE"
echo "  Stage 1 LR: $STAGE1_LR (scaled for larger batch)"
echo "  Stage 2 Epochs: $STAGE2_EPOCHS"
echo "  Stage 2 Batch Size: $STAGE2_BATCH_SIZE"
echo "  Stage 2 LR: $STAGE2_LR"
echo "  Device: $DEVICE"
echo ""
echo "Note: This requires ~16-20GB GPU memory"
echo ""

# Create output directories
mkdir -p checkpoints/stage1_bs16
mkdir -p checkpoints/stage2_bs16

# Run training
cd src

python train.py \
    --model_name bert-base-uncased \
    --stage1_dataset $STAGE1_DATASET \
    --stage1_epochs $STAGE1_EPOCHS \
    --stage1_batch_size $STAGE1_BATCH_SIZE \
    --stage1_lr $STAGE1_LR \
    --stage1_output_dir ../checkpoints/stage1_bs16 \
    --stage2_epochs $STAGE2_EPOCHS \
    --stage2_batch_size $STAGE2_BATCH_SIZE \
    --stage2_lr $STAGE2_LR \
    --stage2_output_dir ../checkpoints/stage2_bs16 \
    --device $DEVICE \
    --seed $SEED

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: checkpoints/stage2_bs16"
echo ""
echo "To evaluate the model, run:"
echo "  cd src && python evaluate.py --model_path ../checkpoints/stage2_bs16 --batch_size 16"
echo ""
