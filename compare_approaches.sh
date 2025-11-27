#!/bin/bash

# Script to compare baseline vs two-stage fine-tuning

echo "=========================================="
echo "Comparing Training Approaches"
echo "=========================================="
echo ""

# Configuration
EPOCHS=3  # Fewer epochs for quick comparison
BATCH_SIZE=8
DEVICE="cuda"

echo "This script will train and compare:"
echo "  1. Baseline: Direct fine-tuning on CommonsenseQA"
echo "  2. Two-stage: SWAG → CommonsenseQA"
echo ""

# Train baseline
echo "=========================================="
echo "Training Baseline Model"
echo "=========================================="

cd src
python train.py \
    --skip_stage1 \
    --stage2_epochs $EPOCHS \
    --stage2_batch_size $BATCH_SIZE \
    --stage2_output_dir ../checkpoints/baseline \
    --device $DEVICE

echo ""
echo "Evaluating baseline..."
python evaluate.py \
    --model_path ../checkpoints/baseline \
    --dataset commonsenseqa \
    --split validation \
    --device $DEVICE

# Train two-stage
echo ""
echo "=========================================="
echo "Training Two-Stage Model (SWAG → CommonsenseQA)"
echo "=========================================="

python train.py \
    --stage1_dataset swag \
    --stage1_epochs $EPOCHS \
    --stage1_batch_size $BATCH_SIZE \
    --stage1_output_dir ../checkpoints/two_stage_stage1 \
    --stage2_epochs $EPOCHS \
    --stage2_batch_size $BATCH_SIZE \
    --stage2_output_dir ../checkpoints/two_stage \
    --device $DEVICE

echo ""
echo "Evaluating two-stage model..."
python evaluate.py \
    --model_path ../checkpoints/two_stage \
    --dataset commonsenseqa \
    --split validation \
    --device $DEVICE

echo ""
echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Baseline: checkpoints/baseline/eval_results_validation.json"
echo "  Two-stage: checkpoints/two_stage/eval_results_validation.json"
echo ""
