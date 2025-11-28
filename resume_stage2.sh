#!/bin/bash

# Resume Stage 2 training from existing Stage 1 checkpoint
# Use this if Stage 1 completed but Stage 2 is stuck at low accuracy

echo "=========================================="
echo "Resuming Stage 2 Training"
echo "=========================================="
echo ""

# Configuration
STAGE1_CHECKPOINT="checkpoints/stage1"  # Your existing Stage 1 model
STAGE2_EPOCHS=8  # For max accuracy
STAGE2_BATCH_SIZE=4  # For max accuracy (change to 8 for default)
STAGE2_LR=8e-6  # For max accuracy (change to 1e-5 for default)
DEVICE="cuda"
SEED=42

echo "This script will:"
echo "  1. Use your existing Stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo "  2. Properly reinitialize the classifier for 5 choices (not 4!)"
echo "  3. Train Stage 2 on CommonsenseQA"
echo ""

# Check if Stage 1 checkpoint exists
if [ ! -d "$STAGE1_CHECKPOINT" ]; then
    echo "❌ Error: Stage 1 checkpoint not found at $STAGE1_CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -d checkpoints/*/ 2>/dev/null || echo "  No checkpoints found"
    echo ""
    echo "Please update STAGE1_CHECKPOINT in this script to point to your Stage 1 model"
    exit 1
fi

echo "✓ Found Stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo ""

# Create output directory
mkdir -p checkpoints/stage2_fixed

# Run Stage 2 training
cd src

echo "Starting Stage 2 training (CommonsenseQA)..."
echo "Note: You should see 'Keeping BERT weights, reinitializing classifier for 5 choices'"
echo ""

python train.py \
    --skip_stage1 \
    --stage1_checkpoint ../$STAGE1_CHECKPOINT \
    --stage2_epochs $STAGE2_EPOCHS \
    --stage2_batch_size $STAGE2_BATCH_SIZE \
    --stage2_lr $STAGE2_LR \
    --stage2_output_dir ../checkpoints/stage2_fixed \
    --device $DEVICE \
    --seed $SEED

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: checkpoints/stage2_fixed"
echo ""
echo "Expected results:"
echo "  - Epoch 1: ~60% accuracy"
echo "  - Epoch 5: ~66% accuracy"
echo "  - Epoch 8: ~67-69% accuracy"
echo ""
echo "To check results:"
echo "  bash check_results.sh"
echo ""
