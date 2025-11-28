#!/bin/bash

# Maximum Accuracy Training for RTX 3090
# Optimized for best possible performance on CommonsenseQA
# Uses batch size 4 with gradient accumulation for stable training

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Error: Python not found"
    exit 1
fi

echo "=========================================="
echo "BERT Fine-tuning - Maximum Accuracy Mode"
echo "=========================================="
echo ""
echo "Hardware: RTX 3090 (24GB)"
echo "Goal: Best possible accuracy"
echo "Strategy: Small batch + gradient accumulation + more epochs"
echo ""

# Configuration optimized for accuracy
STAGE1_DATASET="swag"  # SWAG is more stable than HellaSwag
STAGE1_EPOCHS=4  # More epochs for thorough learning
STAGE1_BATCH_SIZE=4  # Small batch for better generalization
STAGE1_LR=1.5e-5  # Conservative LR for stable training

STAGE2_EPOCHS=8  # Plenty of epochs for CommonsenseQA
STAGE2_BATCH_SIZE=4  # Small batch for best generalization
STAGE2_LR=8e-6  # Lower LR for careful fine-tuning

DEVICE="cuda"
SEED=42

echo "Configuration:"
echo "  Stage 1 Dataset: $STAGE1_DATASET (most stable)"
echo "  Stage 1: $STAGE1_EPOCHS epochs, batch $STAGE1_BATCH_SIZE, LR $STAGE1_LR"
echo "  Stage 2: $STAGE2_EPOCHS epochs, batch $STAGE2_BATCH_SIZE, LR $STAGE2_LR"
echo "  Memory usage: ~5-6GB (very safe for RTX 3090)"
echo ""
echo "Expected results:"
echo "  - Training time: ~10-12 hours total"
echo "  - Best generalization (small batch regularization)"
echo "  - Target accuracy: 67-69% on CommonsenseQA"
echo ""

# Create output directories
mkdir -p checkpoints/max_accuracy_stage1
mkdir -p checkpoints/max_accuracy_final

# Run training
cd src

echo "Starting Stage 1: SWAG fine-tuning..."
echo ""

$PYTHON train.py \
    --model_name bert-base-uncased \
    --max_length 256 \
    --stage1_dataset $STAGE1_DATASET \
    --stage1_epochs $STAGE1_EPOCHS \
    --stage1_batch_size $STAGE1_BATCH_SIZE \
    --stage1_lr $STAGE1_LR \
    --stage1_output_dir ../checkpoints/max_accuracy_stage1 \
    --stage2_epochs $STAGE2_EPOCHS \
    --stage2_batch_size $STAGE2_BATCH_SIZE \
    --stage2_lr $STAGE2_LR \
    --stage2_output_dir ../checkpoints/max_accuracy_final \
    --device $DEVICE \
    --seed $SEED

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Final model: checkpoints/max_accuracy_final"
echo ""
echo "Evaluating on validation set..."
$PYTHON evaluate.py \
    --model_path ../checkpoints/max_accuracy_final \
    --dataset commonsenseqa \
    --split validation \
    --batch_size 16 \
    --device $DEVICE

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Check validation results above"
echo "2. If satisfied, you can also evaluate on:"
echo "   - SWAG: python evaluate.py --model_path ../checkpoints/max_accuracy_stage1 --dataset swag"
echo "   - HellaSwag: python evaluate.py --model_path ../checkpoints/max_accuracy_final --dataset hellaswag"
echo ""
echo "3. For inference on custom questions:"
echo "   - Edit src/inference.py to use model_path = '../checkpoints/max_accuracy_final'"
echo "   - Run: python inference.py"
echo ""
